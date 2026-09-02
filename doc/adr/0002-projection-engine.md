# ADR-0002: Projection engine

Status: accepted. Video sources use one FFmpeg decode per bounded view batch;
image sources are decoded once per frame. A shared `ViewSpec` drives preview,
media export, rigs and datasets. Projection maps are cached with bounded memory.
