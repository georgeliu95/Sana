// Two-Spark exchange over RDMA from host-pinned buffers (the DGX Spark porting guide's no-GPUDirect fallback), without
// NCCL: no GPU kernel, no staging copy. Python allocates one send and one receive buffer (cudaHostAlloc) and a flag
// array in host-pinned memory. A slot is a region of both buffers (my send region goes to the same region of the
// peer's receive buffer); slot 0 is the whole buffer unless sr_slot redefines it. Per slot, the GPU stream writes
// flags[2 * slot] = seq when the send region is ready (cuStreamWriteValue32) and waits for flags[2 * slot + 1] >= seq
// (cuStreamWaitValue32). The progress thread, per slot:
//   - once flags[2 * slot] = seq, sends the peer READY(slot, seq) (a zero-length SEND WITH IMMEDIATE, bit 31 set): the
//     stream only gets there after its consumers of the slot's previous exchange, so its receive region is free;
//   - writes the send region as CHUNK-sized RDMA WRITE WITH IMMEDIATE (imm = slot << 24 | seq) spread over both RoCE
//     ports and QPS queue pairs per port, only after the peer's READY(slot, seq);
//   - sets flags[2 * slot + 1] = seq once its writes completed and all of the peer's arrived.
// One exchange in flight per slot; my writes go out one slot at a time, lowest ready slot first (a pipeline's
// earliest-needed data first). Rank 0 listens (rdma_cm) on its port addresses, rank 1 connects; each side's receive
// buffer address and rkey travel in the connection's private data.
//
// API (ctypes): h = sr_open(rank, "ip0,ip1" local, "ip0,ip1" peer, tcp_port, qps, send, recv, bytes, chunk, flags)
//               sr_slot(h, slot, send_off, recv_off, bytes) (before sr_connect; same layout on both ranks),
//               sr_connect(h) -> 0 (rank 0 blocks until rank 1 connected; call after a barrier), sr_error(h),
//               sr_early(h), sr_close(h). SR_NO_FLOW=1 (test only) writes without waiting for READY.
//               Built on first use by transport.py (gcc, libibverbs, librdmacm); measured in h3-unified-20261027
//               (README "Step 2", "Step 3") as bench/hostmem/spark_rdma.c.
#include <arpa/inet.h>
#include <errno.h>
#include <pthread.h>
#include <rdma/rdma_cma.h>
#include <rdma/rdma_verbs.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define MAX_PORTS 2
#define MAX_QPS 8
#define MAX_SLOTS 64
#define RECV_DEPTH 128
#define SEQS 4
#define READY_BIT 0x80000000u
#define SEQ_MASK 0xffffffu

struct peer_info { uint64_t addr; uint32_t rkey; uint32_t pad; };

struct port {
    char local[64], peer[64];
    struct rdma_cm_id *listen_id, *ids[MAX_QPS];
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_mr *mr_send, *mr_recv;
    struct peer_info remote;
};

struct slot {
    size_t send_off, recv_off, bytes, nchunks;
    uint32_t posted, done, ready_sent, peer_ready;
    size_t sent[SEQS], recvd[SEQS];
    int defined;
};

struct sr {
    int rank, nports, qps, tcp_port, nslots;
    char *send, *recv;
    size_t bytes, chunk;
    volatile uint32_t *flags;
    struct port ports[MAX_PORTS];
    struct slot slots[MAX_SLOTS];
    pthread_t thread;
    volatile int stop, error, running;
    volatile uint32_t early;  // exchanges whose data went out before the peer's READY (SR_NO_FLOW only)
};

#define FAIL(...) do { fprintf(stderr, "spark_rdma: " __VA_ARGS__); fprintf(stderr, " (errno %d %s)\n", errno, strerror(errno)); return -1; } while (0)

static int post_recvs(struct ibv_qp *qp, uint64_t tag, int n) {
    for (int i = 0; i < n; i++) {
        struct ibv_recv_wr wr = {.wr_id = tag, .sg_list = NULL, .num_sge = 0}, *bad;
        if (ibv_post_recv(qp, &wr, &bad)) return -1;
    }
    return 0;
}

static int setup_port_resources(struct sr *s, struct port *p, struct ibv_context *ctx) {
    if (p->pd) return 0;
    p->pd = ibv_alloc_pd(ctx);
    p->cq = ibv_create_cq(ctx, 8192, NULL, NULL, 0);
    if (!p->pd || !p->cq) FAIL("pd/cq on %s", p->local);
    p->mr_send = ibv_reg_mr(p->pd, s->send, s->bytes, IBV_ACCESS_LOCAL_WRITE);
    p->mr_recv = ibv_reg_mr(p->pd, s->recv, s->bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!p->mr_send || !p->mr_recv) FAIL("ibv_reg_mr on %s", p->local);
    return 0;
}

static int create_qp(struct port *p, struct rdma_cm_id *id, uint64_t tag) {
    struct ibv_qp_init_attr attr;
    memset(&attr, 0, sizeof attr);
    attr.send_cq = attr.recv_cq = p->cq;
    attr.cap.max_send_wr = 512;
    attr.cap.max_recv_wr = RECV_DEPTH;
    attr.cap.max_send_sge = attr.cap.max_recv_sge = 1;
    attr.qp_type = IBV_QPT_RC;
    if (rdma_create_qp(id, p->pd, &attr)) FAIL("rdma_create_qp on %s", p->local);
    if (post_recvs(id->qp, tag, RECV_DEPTH)) FAIL("post_recv on %s", p->local);
    return 0;
}

static void conn_param(struct rdma_conn_param *cp, struct peer_info *mine) {
    memset(cp, 0, sizeof *cp);
    cp->private_data = mine;
    cp->private_data_len = sizeof *mine;
    cp->responder_resources = cp->initiator_depth = 1;
    cp->retry_count = 7;
    cp->rnr_retry_count = 7;
}

int sr_slot(void *h, int slot, size_t send_off, size_t recv_off, size_t bytes) {
    struct sr *s = h;
    if (slot < 0 || slot >= MAX_SLOTS || s->running || !bytes || send_off + bytes > s->bytes || recv_off + bytes > s->bytes)
        return -1;
    struct slot *sl = &s->slots[slot];
    memset(sl, 0, sizeof *sl);
    sl->send_off = send_off; sl->recv_off = recv_off; sl->bytes = bytes;
    sl->nchunks = (bytes + s->chunk - 1) / s->chunk;
    sl->defined = 1;
    if (slot >= s->nslots) s->nslots = slot + 1;
    return 0;
}

void *sr_open(int rank, const char *local, const char *peer, int tcp_port, int qps, void *send, void *recv,
              size_t bytes, size_t chunk, volatile uint32_t *flags) {
    struct sr *s = calloc(1, sizeof *s);
    s->rank = rank; s->qps = qps < MAX_QPS ? qps : MAX_QPS; s->tcp_port = tcp_port;
    s->send = send; s->recv = recv; s->bytes = bytes; s->chunk = chunk; s->flags = flags;
    sr_slot(s, 0, 0, 0, bytes);
    char lbuf[256], pbuf[256], *sl, *sp;
    snprintf(lbuf, sizeof lbuf, "%s", local);
    snprintf(pbuf, sizeof pbuf, "%s", peer);
    char *l = strtok_r(lbuf, ",", &sl), *q = strtok_r(pbuf, ",", &sp);
    while (l && q && s->nports < MAX_PORTS) {
        snprintf(s->ports[s->nports].local, 64, "%s", l);
        snprintf(s->ports[s->nports].peer, 64, "%s", q);
        s->nports++;
        l = strtok_r(NULL, ",", &sl); q = strtok_r(NULL, ",", &sp);
    }
    if (rank == 0) {  // listen now, accept in sr_connect
        for (int i = 0; i < s->nports; i++) {
            struct port *p = &s->ports[i];
            char portstr[16];
            snprintf(portstr, sizeof portstr, "%d", tcp_port + i);
            struct rdma_addrinfo hints, *res;
            memset(&hints, 0, sizeof hints);
            hints.ai_flags = RAI_PASSIVE;
            hints.ai_port_space = RDMA_PS_TCP;
            if (rdma_getaddrinfo(p->local, portstr, &hints, &res) || rdma_create_ep(&p->listen_id, res, NULL, NULL) ||
                rdma_listen(p->listen_id, s->qps) || setup_port_resources(s, p, p->listen_id->verbs)) {
                fprintf(stderr, "spark_rdma: listen on %s:%s failed (errno %d %s)\n", p->local, portstr, errno, strerror(errno));
                free(s);
                return NULL;
            }
            rdma_freeaddrinfo(res);
        }
    }
    return s;
}

static int connect_all(struct sr *s) {
    for (int i = 0; i < s->nports; i++) {
        struct port *p = &s->ports[i];
        for (int k = 0; k < s->qps; k++) {
            struct rdma_conn_param cp;
            struct peer_info mine;
            uint64_t tag = (uint64_t)(i * MAX_QPS + k);
            if (s->rank == 0) {
                struct rdma_cm_id *id;
                if (rdma_get_request(p->listen_id, &id)) FAIL("rdma_get_request on %s", p->local);
                if (create_qp(p, id, tag)) return -1;
                memcpy(&p->remote, id->event->param.conn.private_data, sizeof p->remote);
                mine = (struct peer_info){(uint64_t)(uintptr_t)s->recv, p->mr_recv->rkey, 0};
                conn_param(&cp, &mine);
                if (rdma_accept(id, &cp)) FAIL("rdma_accept on %s", p->local);
                p->ids[k] = id;
            } else {
                char portstr[16];
                snprintf(portstr, sizeof portstr, "%d", s->tcp_port + i);
                struct rdma_addrinfo hints, *res;
                memset(&hints, 0, sizeof hints);
                hints.ai_port_space = RDMA_PS_TCP;
                struct rdma_cm_id *id = NULL;
                for (int tries = 0;; tries++) {  // the listener may not be up yet
                    if (rdma_getaddrinfo(p->peer, portstr, &hints, &res)) FAIL("rdma_getaddrinfo %s", p->peer);
                    if (rdma_create_ep(&id, res, NULL, NULL)) FAIL("rdma_create_ep %s", p->peer);
                    rdma_freeaddrinfo(res);
                    if (setup_port_resources(s, p, id->verbs) || create_qp(p, id, tag)) return -1;
                    mine = (struct peer_info){(uint64_t)(uintptr_t)s->recv, p->mr_recv->rkey, 0};
                    conn_param(&cp, &mine);
                    if (!rdma_connect(id, &cp)) break;
                    rdma_destroy_qp(id);
                    rdma_destroy_ep(id);
                    if (tries > 300) FAIL("rdma_connect %s:%s", p->peer, portstr);
                    usleep(100000);
                }
                memcpy(&p->remote, id->event->param.conn.private_data, sizeof p->remote);
                p->ids[k] = id;
            }
        }
    }
    return 0;
}

static int post_exchange(struct sr *s, int slot, uint32_t seq) {
    struct slot *sl = &s->slots[slot];
    for (size_t c = 0; c < sl->nchunks; c++) {
        size_t lane = c + (size_t)slot;  // spread small slots over ports and QPs
        int pi = lane % s->nports, qi = (lane / s->nports) % s->qps;
        struct port *p = &s->ports[pi];
        size_t off = c * s->chunk, len = sl->bytes - off < s->chunk ? sl->bytes - off : s->chunk;
        struct ibv_sge sge = {.addr = (uint64_t)(uintptr_t)(s->send + sl->send_off + off), .length = (uint32_t)len,
                              .lkey = p->mr_send->lkey};
        struct ibv_send_wr wr, *bad;
        memset(&wr, 0, sizeof wr);
        wr.wr_id = ((uint64_t)slot << 56) | ((uint64_t)(seq & SEQ_MASK) << 32) | c;
        wr.sg_list = &sge;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
        wr.send_flags = IBV_SEND_SIGNALED;
        wr.imm_data = htonl(((uint32_t)slot << 24) | (seq & SEQ_MASK));
        wr.wr.rdma.remote_addr = p->remote.addr + sl->recv_off + off;
        wr.wr.rdma.rkey = p->remote.rkey;
        if (ibv_post_send(p->ids[qi]->qp, &wr, &bad)) FAIL("ibv_post_send slot %d seq %u chunk %zu", slot, seq, c);
    }
    return 0;
}

static int post_ready(struct sr *s, int slot, uint32_t seq) {
    struct ibv_send_wr wr, *bad;
    memset(&wr, 0, sizeof wr);
    wr.wr_id = ~0ull;
    wr.opcode = IBV_WR_SEND_WITH_IMM;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.imm_data = htonl(READY_BIT | ((uint32_t)slot << 24) | (seq & SEQ_MASK));
    if (ibv_post_send(s->ports[0].ids[0]->qp, &wr, &bad)) FAIL("ibv_post_send READY slot %d seq %u", slot, seq);
    return 0;
}

static void *progress(void *arg) {
    struct sr *s = arg;
    struct ibv_wc wc[64];
    int inflight = -1;  // slot whose writes are on the wire
    int no_flow = getenv("SR_NO_FLOW") != NULL;
    if (no_flow) fprintf(stderr, "spark_rdma: rank %d SR_NO_FLOW set, no flow control\n", s->rank);
    while (!s->stop) {
        int busy = inflight >= 0;
        for (int i = 0; i < s->nslots; i++) {
            struct slot *sl = &s->slots[i];
            if (!sl->defined) continue;
            uint32_t want = __atomic_load_n(&s->flags[2 * i], __ATOMIC_ACQUIRE);
            if (want > sl->ready_sent) {
                if (post_ready(s, i, sl->ready_sent + 1)) { s->error = 1; goto out; }
                sl->ready_sent++;
            }
            if (want > sl->done) busy = 1;
            if (inflight < 0 && want > sl->posted && sl->posted == sl->done && (sl->peer_ready > sl->posted || no_flow)) {
                if (sl->peer_ready <= sl->posted) s->early++;
                if (post_exchange(s, i, sl->posted + 1)) { s->error = 1; goto out; }
                sl->posted++;
                inflight = i;
            }
        }
        for (int p = 0; p < s->nports; p++) {
            int n = ibv_poll_cq(s->ports[p].cq, 64, wc);
            for (int j = 0; j < n; j++) {
                busy = 1;
                if (wc[j].status != IBV_WC_SUCCESS) {
                    fprintf(stderr, "spark_rdma: completion error %s (opcode %d)\n", ibv_wc_status_str(wc[j].status), wc[j].opcode);
                    s->error = 1;
                    goto out;
                }
                if (wc[j].opcode == IBV_WC_RDMA_WRITE) {
                    int slot = (int)(wc[j].wr_id >> 56);
                    uint32_t seq = (uint32_t)(wc[j].wr_id >> 32) & SEQ_MASK;
                    struct slot *sl = &s->slots[slot];
                    if (++sl->sent[seq % SEQS] == sl->nchunks && slot == inflight) inflight = -1;
                } else if (wc[j].opcode == IBV_WC_RECV_RDMA_WITH_IMM || wc[j].opcode == IBV_WC_RECV) {
                    uint32_t imm = ntohl(wc[j].imm_data);
                    struct slot *sl = &s->slots[(imm >> 24) & 0x7f];
                    uint32_t seq = imm & SEQ_MASK;
                    if (imm & READY_BIT) {
                        if (seq > sl->peer_ready) sl->peer_ready = seq;
                    } else {
                        sl->recvd[seq % SEQS]++;
                    }
                    uint64_t tag = wc[j].wr_id;
                    if (post_recvs(s->ports[tag / MAX_QPS].ids[tag % MAX_QPS]->qp, tag, 1)) { s->error = 1; goto out; }
                }
            }
        }
        for (int i = 0; i < s->nslots; i++) {
            struct slot *sl = &s->slots[i];
            uint32_t cur = sl->done + 1;
            if (sl->defined && sl->posted >= cur && sl->sent[cur % SEQS] == sl->nchunks && sl->recvd[cur % SEQS] >= sl->nchunks) {
                sl->sent[cur % SEQS] = 0;
                sl->recvd[cur % SEQS] -= sl->nchunks;
                sl->done = cur;
                __atomic_store_n(&s->flags[2 * i + 1], cur, __ATOMIC_RELEASE);
            }
        }
        if (!busy) {  // nothing requested or on the wire: do not hold a core between exchanges
            struct timespec ts = {0, 10000};
            nanosleep(&ts, NULL);
        }
    }
out:
    if (s->error)  // release every stream waiting on a slot; the caller checks sr_error
        for (int i = 0; i < s->nslots; i++) __atomic_store_n(&s->flags[2 * i + 1], 0xffffffffu, __ATOMIC_RELEASE);
    return NULL;
}

int sr_connect(void *h) {
    struct sr *s = h;
    if (connect_all(s)) return -1;
    s->running = 1;
    if (pthread_create(&s->thread, NULL, progress, s)) { s->running = 0; return -1; }
    return 0;
}

int sr_error(void *h) { return ((struct sr *)h)->error; }

unsigned sr_early(void *h) { return ((struct sr *)h)->early; }

void sr_close(void *h) {
    struct sr *s = h;
    if (!s) return;
    s->stop = 1;
    if (s->running) pthread_join(s->thread, NULL);
    for (int i = 0; i < s->nports; i++) {
        struct port *p = &s->ports[i];
        for (int k = 0; k < s->qps; k++)
            if (p->ids[k]) { rdma_disconnect(p->ids[k]); rdma_destroy_qp(p->ids[k]); rdma_destroy_ep(p->ids[k]); }
        if (p->mr_send) ibv_dereg_mr(p->mr_send);
        if (p->mr_recv) ibv_dereg_mr(p->mr_recv);
        if (p->cq) ibv_destroy_cq(p->cq);
        if (p->pd) ibv_dealloc_pd(p->pd);
        if (p->listen_id) rdma_destroy_ep(p->listen_id);
    }
    free(s);
}
