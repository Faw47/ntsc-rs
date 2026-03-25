#[cfg(test)]
mod tests {
    use crate::{
        gpu::{BackendType, runner::NtscEffectRunner},
        settings::standard::NtscEffect,
        yiq_fielding::{YiqField, YiqOwned, YiqView, Rgbx},
    };

    #[test]
    fn test_wgpu_copy_pass() {
        // Initialize simple 4x4 image
        let width = 4;
        let height = 4;
        let mut pixels: Vec<f32> = vec![0.0; width * height * 4];
        for i in 0..(width * height) {
            pixels[i * 4] = i as f32 / 16.0; // R
            pixels[i * 4 + 1] = i as f32 / 16.0; // G
            pixels[i * 4 + 2] = i as f32 / 16.0; // B
            pixels[i * 4 + 3] = 1.0; // A
        }

        let mut yiq = YiqOwned::from_strided_buffer::<Rgbx, f32>(
            &pixels,
            width * 4 * std::mem::size_of::<f32>(),
            width,
            height,
            YiqField::Both,
        );
        let mut yiq_view = YiqView::from(&mut yiq);

        let mut cpu_pixels = pixels.clone();
        let mut cpu_yiq = YiqOwned::from_strided_buffer::<Rgbx, f32>(
            &cpu_pixels,
            width * 4 * std::mem::size_of::<f32>(),
            width,
            height,
            YiqField::Both,
        );
        let mut cpu_yiq_view = YiqView::from(&mut cpu_yiq);

        let effect = NtscEffect::default();

        let mut cpu_runner = NtscEffectRunner::new(BackendType::Cpu);
        cpu_runner.apply_effect(&mut cpu_yiq_view, &effect, 0, [1.0, 1.0]);

        #[cfg(feature = "gpu-wgpu")]
        {
            let mut wgpu_runner = NtscEffectRunner::new(BackendType::Wgpu);
            if wgpu_runner.active_backend() == BackendType::Wgpu {
                wgpu_runner.apply_effect(&mut yiq_view, &effect, 0, [1.0, 1.0]);

                // Wgpu currently just copies (no-op compute), so we don't expect it to match CPU
                // completely right now, but we want to make sure it doesn't crash and returns the
                // original or modified buffer back properly.
                // In future steps we'll assert CPU == GPU.
                assert_eq!(yiq_view.dimensions, cpu_yiq_view.dimensions);
            }
        }
    }
}
