"""Writer views preserving original decoder checks and the native encoder."""


def tracked_writer_chunks(chunks, receipt, *, expected_dtype):
    """Use the gated 16-frame slicing expression without copies or casts.

    Exhaust the original iterator so its finite/frame-count checks still run.
    Receipt contains CPU metadata only, including partial counts on failure.
    """
    source_frames = 0
    for chunk in chunks:
        if (chunk.ndim != 4 or tuple(chunk.shape[1:]) != (768, 1344, 3)
                or chunk.dtype != expected_dtype):
            raise RuntimeError("writer chunk16 requires matched BF16 768x1344 RGB")
        receipt["source_chunks"] += 1
        receipt["source_layouts"].append({"shape": list(chunk.shape), "dtype": str(chunk.dtype),
                                          "stride": list(chunk.stride())})
        source_frames += int(chunk.shape[0])
        remaining = max(0, 124 - receipt["frames"])
        for start in range(0, min(chunk.shape[0], remaining), 16):
            view = chunk[start:min(start + 16, remaining)]
            receipt["emitted_chunks"] += 1
            receipt["frames"] += int(view.shape[0])
            yield view
    receipt["decoded_frames"] = source_frames
    if source_frames != 129:
        raise RuntimeError(f"decoder yielded {source_frames} frames, expected129")
    if receipt["frames"] != 124:
        raise RuntimeError(f"writer chunk16 consumed {receipt['frames']} frames, expected 124")
    receipt["complete"] = True
