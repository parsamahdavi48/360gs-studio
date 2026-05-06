from gui.steps.step4_cubemap import is_spheresfm_rtx50_cuda_error_line


def test_spheresfm_rtx50_diagnostic_detects_no_kernel_image_error() -> None:
    assert is_spheresfm_rtx50_cuda_error_line(
        "CuTexImage::BindTexture:\tno kernel image is available for execution on the device"
    )
    assert is_spheresfm_rtx50_cuda_error_line(
        "PyramidCU::GenerateFeatureList: no kernel image is available for execution on the device"
    )


def test_spheresfm_rtx50_diagnostic_detects_related_invalid_device_function() -> None:
    assert is_spheresfm_rtx50_cuda_error_line("PyramidCU::GenerateFeatureList: invalid device function")


def test_spheresfm_rtx50_diagnostic_ignores_unrelated_lines() -> None:
    assert not is_spheresfm_rtx50_cuda_error_line("Processed file [46/467]")
    assert not is_spheresfm_rtx50_cuda_error_line("ERROR: Failed to extract features.")
