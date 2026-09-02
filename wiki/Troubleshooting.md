# Troubleshooting

## A tool is unavailable

Run `360gs-studio doctor`. Optional features remain disabled until their individual capability checks pass.

## A path is rejected

Use a short path containing ASCII letters and numbers, for example `D:\work\scene01`. Some image and SfM tools fail on control characters, quotation marks, very long paths, or non-ASCII paths.

## NVIDIA encoding is unavailable

NVENC is enabled only when the selected FFmpeg build reports a compatible encoder. Use CPU encoding when the probe fails.

## A job was interrupted

Open the project again. Persisted jobs that were running during a crash are marked `interrupted`; partial staging output is not presented as a complete dataset.

## Reporting a problem

Use the appropriate GitHub issue form and attach the diagnostic summary and relevant log. Never include passwords, access tokens, or private capture files.
