//! ntsc-rs Design System
//!
//! A custom theme layer for egui implementing a "phosphor monitor" aesthetic.
//! Provides semantic color tokens, typography helpers, frame constructors,
//! status tones, and component utilities.

use eframe::egui::{self, Color32, FontId, Margin, RichText, Shadow, Stroke, TextStyle, Ui};

// ──────────────────────────────────────────────
// ACCENT
// ──────────────────────────────────────────────

/// Primary accent — a cool, desaturated phosphor blue.
const PHOSPHOR_BLUE: Color32 = Color32::from_rgb(86, 156, 214);
const PHOSPHOR_BLUE_HOVER: Color32 = Color32::from_rgb(106, 176, 234);
const PHOSPHOR_BLUE_DIM: Color32 = Color32::from_rgb(66, 126, 180);

// ──────────────────────────────────────────────
// PALETTE — semantic color tokens
// ──────────────────────────────────────────────

/// Semantic color palette. One instance per theme (dark / light).
#[derive(Clone, Debug)]
pub struct ThemePalette {
    // Surfaces — 5-tier hierarchy from deepest to topmost
    pub surface_sunken: Color32,
    pub app_bg: Color32,
    pub surface_base: Color32,
    pub surface_raised: Color32,
    pub surface_overlay: Color32,

    // Lines
    pub line_soft: Color32,
    pub line_strong: Color32,

    // Text
    pub text_primary: Color32,
    pub text_secondary: Color32,
    pub text_muted: Color32,

    // Accent
    pub accent: Color32,
    pub accent_hover: Color32,
    pub accent_dim: Color32,
    pub accent_text: Color32,

    // Status tones
    pub tone_info: Color32,
    pub tone_success: Color32,
    pub tone_warning: Color32,
    pub tone_danger: Color32,
}

fn dark_palette() -> ThemePalette {
    ThemePalette {
        surface_sunken: Color32::from_rgb(18, 18, 22),
        app_bg: Color32::from_rgb(24, 24, 30),
        surface_base: Color32::from_rgb(32, 32, 40),
        surface_raised: Color32::from_rgb(42, 42, 52),
        surface_overlay: Color32::from_rgb(52, 52, 64),

        line_soft: Color32::from_rgba_premultiplied(255, 255, 255, 18),
        line_strong: Color32::from_rgba_premultiplied(255, 255, 255, 38),

        text_primary: Color32::from_rgb(230, 230, 238),
        text_secondary: Color32::from_rgb(180, 180, 195),
        text_muted: Color32::from_rgb(120, 120, 140),

        accent: PHOSPHOR_BLUE,
        accent_hover: PHOSPHOR_BLUE_HOVER,
        accent_dim: PHOSPHOR_BLUE_DIM,
        accent_text: Color32::from_rgb(255, 255, 255),

        tone_info: Color32::from_rgb(86, 156, 214),
        tone_success: Color32::from_rgb(78, 186, 128),
        tone_warning: Color32::from_rgb(220, 180, 70),
        tone_danger: Color32::from_rgb(220, 80, 80),
    }
}

fn light_palette() -> ThemePalette {
    ThemePalette {
        surface_sunken: Color32::from_rgb(215, 215, 225),
        app_bg: Color32::from_rgb(230, 230, 238),
        surface_base: Color32::from_rgb(242, 242, 248),
        surface_raised: Color32::from_rgb(252, 252, 255),
        surface_overlay: Color32::from_rgb(255, 255, 255),

        line_soft: Color32::from_rgba_premultiplied(0, 0, 0, 15),
        line_strong: Color32::from_rgba_premultiplied(0, 0, 0, 35),

        text_primary: Color32::from_rgb(30, 30, 38),
        text_secondary: Color32::from_rgb(80, 80, 100),
        text_muted: Color32::from_rgb(130, 130, 150),

        accent: Color32::from_rgb(40, 110, 180),
        accent_hover: Color32::from_rgb(50, 130, 200),
        accent_dim: Color32::from_rgb(30, 90, 150),
        accent_text: Color32::WHITE,

        tone_info: Color32::from_rgb(40, 110, 180),
        tone_success: Color32::from_rgb(40, 150, 90),
        tone_warning: Color32::from_rgb(190, 140, 30),
        tone_danger: Color32::from_rgb(200, 50, 50),
    }
}

/// Get palette for the current visuals (dark or light).
pub fn palette(ctx: &egui::Context) -> ThemePalette {
    if ctx.style().visuals.dark_mode {
        dark_palette()
    } else {
        light_palette()
    }
}

/// Get palette from visuals directly (useful inside widgets without Context).
pub fn palette_for_visuals(visuals: &egui::Visuals) -> ThemePalette {
    if visuals.dark_mode {
        dark_palette()
    } else {
        light_palette()
    }
}

// ──────────────────────────────────────────────
// STYLE CONFIGURATION
// ──────────────────────────────────────────────

/// Apply the full design system to an egui context. Call once at startup.
pub fn configure_style(ctx: &egui::Context) {
    ctx.style_mut(|style| {
        // ── TYPOGRAPHY ──
        style.text_styles = [
            (TextStyle::Heading, FontId::proportional(22.0)),
            (TextStyle::Body, FontId::proportional(14.0)),
            (TextStyle::Button, FontId::proportional(13.5)),
            (TextStyle::Small, FontId::proportional(11.5)),
            (TextStyle::Monospace, FontId::monospace(13.5)),
            (TextStyle::Name("Section".into()), FontId::proportional(16.0)),
            (TextStyle::Name("Label".into()), FontId::proportional(12.5)),
            (TextStyle::Name("Overline".into()), FontId::proportional(11.0)),
        ]
        .into();

        // ── SPACING ──
        style.spacing.item_spacing = egui::vec2(8.0, 6.0);
        style.spacing.button_padding = egui::vec2(12.0, 6.0);
        style.spacing.window_margin = Margin::same(12);
        style.spacing.menu_margin = Margin::same(8);
        style.spacing.interact_size.y = 28.0;
        style.spacing.indent = 16.0;

        // ── CORNER RADII ──
        style.visuals.window_corner_radius = egui::CornerRadius::same(8);
        style.visuals.menu_corner_radius = egui::CornerRadius::same(8);
        style.visuals.widgets.noninteractive.corner_radius = egui::CornerRadius::same(4);
        style.visuals.widgets.inactive.corner_radius = egui::CornerRadius::same(4);
        style.visuals.widgets.hovered.corner_radius = egui::CornerRadius::same(4);
        style.visuals.widgets.active.corner_radius = egui::CornerRadius::same(4);
        style.visuals.widgets.open.corner_radius = egui::CornerRadius::same(4);

        // ── INTERACTIONS ──
        style.interaction.tooltip_delay = 0.5;
        style.interaction.show_tooltips_only_when_still = false;

        // ── SHADOWS ──
        style.visuals.popup_shadow = Shadow {
            offset: [0, 4],
            blur: 12,
            spread: 0,
            color: Color32::from_black_alpha(60),
        };
        style.visuals.window_shadow = Shadow {
            offset: [0, 6],
            blur: 20,
            spread: 2,
            color: Color32::from_black_alpha(50),
        };
    });

    // Apply themed colors to whatever the current theme is
    apply_themed_visuals(ctx);
}

/// Apply palette-specific colors. Called on theme change and at startup.
pub fn apply_themed_visuals(ctx: &egui::Context) {
    let pal = palette(ctx);

    ctx.style_mut(|style| {
        let vis = &mut style.visuals;
        vis.override_text_color = Some(pal.text_primary);
        vis.panel_fill = pal.app_bg;
        vis.window_fill = pal.surface_raised;
        vis.extreme_bg_color = pal.surface_sunken;
        vis.faint_bg_color = pal.surface_base;
        vis.code_bg_color = pal.surface_sunken;

        // Selection
        vis.selection.bg_fill = pal.accent.gamma_multiply(0.35);
        vis.selection.stroke = Stroke::new(1.0, pal.accent);

        // Hyperlink
        vis.hyperlink_color = pal.accent;

        // ── WIDGET STATES ──
        // Noninteractive (labels, separators)
        vis.widgets.noninteractive.bg_fill = pal.surface_base;
        vis.widgets.noninteractive.bg_stroke = Stroke::new(1.0, pal.line_soft);
        vis.widgets.noninteractive.fg_stroke = Stroke::new(1.0, pal.text_secondary);
        vis.widgets.noninteractive.weak_bg_fill = pal.surface_base;

        // Inactive (buttons not hovered)
        vis.widgets.inactive.bg_fill = pal.surface_raised;
        vis.widgets.inactive.bg_stroke = Stroke::new(1.0, pal.line_soft);
        vis.widgets.inactive.fg_stroke = Stroke::new(1.0, pal.text_primary);
        vis.widgets.inactive.weak_bg_fill = pal.surface_raised;

        // Hovered
        vis.widgets.hovered.bg_fill = pal.surface_overlay;
        vis.widgets.hovered.bg_stroke = Stroke::new(1.0, pal.accent);
        vis.widgets.hovered.fg_stroke = Stroke::new(1.0, pal.text_primary);
        vis.widgets.hovered.weak_bg_fill = pal.surface_overlay;

        // Active (pressed / dragging)
        vis.widgets.active.bg_fill = pal.accent_dim;
        vis.widgets.active.bg_stroke = Stroke::new(1.0, pal.accent);
        vis.widgets.active.fg_stroke = Stroke::new(1.5, pal.text_primary);
        vis.widgets.active.weak_bg_fill = pal.accent_dim;

        // Open (e.g. combobox expanded)
        vis.widgets.open.bg_fill = pal.surface_overlay;
        vis.widgets.open.bg_stroke = Stroke::new(1.0, pal.accent);
        vis.widgets.open.fg_stroke = Stroke::new(1.0, pal.text_primary);
        vis.widgets.open.weak_bg_fill = pal.surface_overlay;

        // Striped background for alternating rows
        vis.striped = true;
    });
}

// ──────────────────────────────────────────────
// TYPOGRAPHY HELPERS
// ──────────────────────────────────────────────

/// Large section heading text.
pub fn section_title(text: impl Into<String>, ui: &Ui) -> RichText {
    let pal = palette_for_visuals(ui.visuals());
    RichText::new(text.into())
        .text_style(TextStyle::Name("Section".into()))
        .strong()
        .color(pal.text_primary)
}

/// Overline label — small uppercase category label.
pub fn overline(text: impl Into<String>, ui: &Ui) -> RichText {
    let pal = palette_for_visuals(ui.visuals());
    RichText::new(text.into().to_uppercase())
        .text_style(TextStyle::Name("Overline".into()))
        .color(pal.text_muted)
}

/// Muted body text — secondary information.
pub fn muted(text: impl Into<String>, ui: &Ui) -> RichText {
    let pal = palette_for_visuals(ui.visuals());
    RichText::new(text.into())
        .text_style(TextStyle::Body)
        .color(pal.text_muted)
}

/// Subtle text — tertiary, nearly invisible.
pub fn subtle(text: impl Into<String>, ui: &Ui) -> RichText {
    let pal = palette_for_visuals(ui.visuals());
    RichText::new(text.into())
        .text_style(TextStyle::Small)
        .color(pal.text_muted.gamma_multiply(0.7))
}

// ──────────────────────────────────────────────
// FRAME CONSTRUCTORS
// ──────────────────────────────────────────────

/// Panel-level frame — used for command bar and major panels.
pub fn panel_frame(ui: &Ui) -> egui::Frame {
    let pal = palette_for_visuals(ui.visuals());
    egui::Frame::NONE
        .fill(pal.surface_base)
        .stroke(Stroke::new(1.0, pal.line_soft))
        .corner_radius(egui::CornerRadius::same(4))
        .inner_margin(Margin::same(8))
}

/// Overlay frame — used for headers/footers inside panels.
pub fn overlay_frame(ui: &Ui) -> egui::Frame {
    let pal = palette_for_visuals(ui.visuals());
    egui::Frame::NONE
        .fill(pal.surface_raised)
        .inner_margin(Margin::symmetric(10, 8))
}

/// Workspace frame — the main video preview background.
pub fn workspace_frame(ui: &Ui) -> egui::Frame {
    let pal = palette_for_visuals(ui.visuals());
    egui::Frame::NONE
        .fill(pal.surface_sunken)
        .inner_margin(Margin::ZERO)
}

/// Dialog frame — floating windows and modals.
pub fn dialog_frame(ctx: &egui::Context) -> egui::Frame {
    let pal = palette(ctx);
    egui::Frame::NONE
        .fill(pal.surface_raised)
        .stroke(Stroke::new(1.5, pal.line_strong))
        .corner_radius(egui::CornerRadius::same(8))
        .inner_margin(Margin::same(16))
        .shadow(Shadow {
            offset: [0, 4],
            blur: 16,
            spread: 2,
            color: Color32::from_black_alpha(40),
        })
}

/// Subtle section frame — used for collapsible settings groups.
pub fn subtle_section_frame(ui: &Ui) -> egui::Frame {
    let pal = palette_for_visuals(ui.visuals());
    egui::Frame::NONE
        .fill(pal.surface_base)
        .stroke(Stroke::new(1.0, pal.line_soft))
        .corner_radius(egui::CornerRadius::same(4))
        .inner_margin(Margin::same(8))
}

// ──────────────────────────────────────────────
// SECTION HELPER
// ──────────────────────────────────────────────

/// Render a titled, optionally-described section with consistent spacing.
/// Returns the inner value (typically `bool` for settings_changed).
pub fn section<R>(
    ui: &mut Ui,
    title: &str,
    description: Option<&str>,
    add_body: impl FnOnce(&mut Ui) -> R,
) -> R {
    ui.add_space(10.0);
    ui.label(section_title(title, ui));
    if let Some(desc) = description {
        ui.label(muted(desc, ui));
    }
    ui.add_space(4.0);
    let result = add_body(ui);
    result
}

// ──────────────────────────────────────────────
// STATUS TONES & BADGES
// ──────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StatusTone {
    Info,
    Success,
    Warning,
    Danger,
}

impl StatusTone {
    fn color(&self, pal: &ThemePalette) -> Color32 {
        match self {
            StatusTone::Info => pal.tone_info,
            StatusTone::Success => pal.tone_success,
            StatusTone::Warning => pal.tone_warning,
            StatusTone::Danger => pal.tone_danger,
        }
    }
}

/// A tinted frame for status messages (error banners, info callouts).
pub fn status_frame(ui: &Ui, tone: StatusTone) -> egui::Frame {
    let pal = palette_for_visuals(ui.visuals());
    let color = tone.color(&pal);
    egui::Frame::NONE
        .fill(color.gamma_multiply(if ui.visuals().dark_mode { 0.12 } else { 0.08 }))
        .stroke(Stroke::new(1.0, color.gamma_multiply(0.4)))
        .corner_radius(egui::CornerRadius::same(4))
        .inner_margin(Margin::symmetric(10, 6))
}

/// Small inline badge — e.g. "1920 × 1080", "Audio", "Rendering".
pub fn status_badge(ui: &mut Ui, text: &str, tone: StatusTone) {
    let pal = palette_for_visuals(ui.visuals());
    let color = tone.color(&pal);

    egui::Frame::NONE
        .fill(color.gamma_multiply(if ui.visuals().dark_mode { 0.15 } else { 0.10 }))
        .stroke(Stroke::new(0.5, color.gamma_multiply(0.4)))
        .corner_radius(egui::CornerRadius::same(3))
        .inner_margin(Margin::symmetric(6, 3))
        .show(ui, |ui| {
            ui.label(
                RichText::new(text)
                    .text_style(TextStyle::Small)
                    .color(color),
            );
        });
}

/// Keyboard shortcut hint badge — e.g. "/" or "⌘1".
pub fn key_hint(ui: &mut Ui, key: &str) {
    let pal = palette_for_visuals(ui.visuals());
    egui::Frame::NONE
        .fill(pal.surface_sunken)
        .stroke(Stroke::new(0.5, pal.line_strong))
        .corner_radius(egui::CornerRadius::same(3))
        .inner_margin(Margin::symmetric(5, 2))
        .show(ui, |ui| {
            ui.label(
                RichText::new(key)
                    .text_style(TextStyle::Small)
                    .color(pal.text_muted),
            );
        });
}

// ──────────────────────────────────────────────
// PRIMARY BUTTON HELPERS
// ──────────────────────────────────────────────

/// Fill color for a primary action button (e.g. "Start Render").
pub fn primary_button_fill(dark_mode: bool) -> Color32 {
    if dark_mode { PHOSPHOR_BLUE } else { Color32::from_rgb(40, 110, 180) }
}

/// Hovered fill for primary button.
pub fn primary_button_fill_hovered(dark_mode: bool) -> Color32 {
    if dark_mode { PHOSPHOR_BLUE_HOVER } else { Color32::from_rgb(50, 130, 200) }
}

/// Text color for primary button.
pub fn primary_button_text_color() -> Color32 {
    Color32::WHITE
}

// ──────────────────────────────────────────────
// CHAPTER DIVIDER
// ──────────────────────────────────────────────

/// A horizontal divider with extra spacing, used between major settings chapters.
pub fn chapter_divider(ui: &mut Ui) {
    ui.add_space(8.0);
    let pal = palette_for_visuals(ui.visuals());
    let rect = ui.available_rect_before_wrap();
    let y = rect.top();
    ui.painter().line_segment(
        [
            egui::pos2(rect.left() + 4.0, y),
            egui::pos2(rect.right() - 4.0, y),
        ],
        Stroke::new(1.0, pal.line_soft),
    );
    ui.add_space(8.0);
}
