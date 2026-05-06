from gui.steps.step4_cubemap import is_colmap_gui_unavailable_output, is_spheresfm_rtx50_cuda_error_line


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


def test_colmap_gui_diagnostic_detects_gui_less_spheresfm_build() -> None:
    assert is_colmap_gui_unavailable_output(
        "ERROR: Cannot start colmap GUI; colmap was built without GUI support or QT dependency is missing."
    )


def test_colmap_gui_diagnostic_ignores_unrelated_colmap_errors() -> None:
    assert not is_colmap_gui_unavailable_output("ERROR: Failed to extract features.")
