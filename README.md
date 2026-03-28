<p align="center">
    <a href="https://ntsc.rs">
        <picture>
            <source media="(prefers-color-scheme: dark)" srcset="./docs/img/logo-darkmode.svg">
            <img alt="ntsc-rs logo" src="./docs/img/logo-lightmode.svg" width="1216">
        </picture>
    </a>
</p>

---

**ntsc-rs** is a video effect which emulates NTSC and VHS video artifacts. It can be used as an After Effects, Premiere, or OpenFX plugin, or as a standalone application.

![Screenshot of the ntsc-rs standalone application](./docs/img/appdemo.png)

## About this fork

This repository is derived from **[ntsc-rs](https://github.com/ntsc-rs/ntsc-rs)**. The original project, its algorithms, and naming remain the work of that upstream team; this fork does not claim ownership of their design or intellectual property.

**Performance.** Processing has been moved onto a **wgpu**-based GPU path. On suitable hardware and typical settings, throughput can reach **up to about six times** that of the original CPU-oriented pipeline in upstream ntsc-rs. Actual speedups depend on GPU, resolution, preset, and system load.

**Pipeline fidelity.** A few effect stages and parameters differ slightly from upstream. The intent is the same nostalgic NTSC/VHS look with faster turnaround, not a pixel-perfect match to every pass in the reference implementation.

**User interface.** The standalone GUI has been reworked for clearer structure and a more deliberate visual treatment than the stock layout.

**How this was built.** Much of the engineering here—including iterative refactors, integration work, and even parts of this README—was carried out with **large language model** assistance, on top of the upstream codebase. The upstream repository remains the authoritative source for the effect model and the original Rust port.

## Download and Install

The latest version of ntsc-rs can be downloaded from [the releases page](https://github.com/valadaptive/ntsc-rs/releases).

After downloading, [read the documentation for how to run it](https://ntsc.rs/docs/standalone-installation/). In particular, ntsc-rs will not work properly on Linux unless you install all of the GStreamer packages listed in the documentation.

## More information

ntsc-rs is a rough Rust port of [ntscqt](https://github.com/JargeZ/ntscqt), a PyQt-based GUI for [ntsc](https://github.com/zhuker/ntsc), itself a Python port of [composite-video-simulator](https://github.com/joncampbell123/composite-video-simulator). Reimplementing the image processing in multithreaded Rust allows it to run at (mostly) real-time speeds.

It's not an exact port--some processing passes have visibly different results, and some new ones have been added.
