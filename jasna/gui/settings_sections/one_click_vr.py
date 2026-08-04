"""Processing-mode controls for standard Jasna and one-click VR."""

from __future__ import annotations

import customtkinter as ctk

from jasna.gui.components import CollapsibleSection, Tooltip
from jasna.gui.locales import t
from jasna.gui.settings_sections.widgets import ValueOptionMenu
from jasna.gui.theme import Colors, Fonts, Sizing


PROCESSING_MODE_STANDARD = "standard"
PROCESSING_MODE_ONE_CLICK_VR = "one_click_vr"


class ProcessingModeSection:
    def __init__(self, parent, widgets: dict, on_modified):
        self._widgets = widgets
        self._on_modified = on_modified
        self._mode_to_label = {
            PROCESSING_MODE_STANDARD: t("processing_mode_standard"),
            PROCESSING_MODE_ONE_CLICK_VR: t("processing_mode_one_click_vr"),
        }
        self._label_to_mode = {
            label: value for value, label in self._mode_to_label.items()
        }

        section = CollapsibleSection(
            parent, t("section_processing_mode"), expanded=True
        )
        section.pack(fill="x", pady=(0, Sizing.PADDING_SMALL))
        content = section.content
        content.configure(corner_radius=Sizing.BORDER_RADIUS)
        inner = ctk.CTkFrame(content, fg_color="transparent")
        inner.pack(fill="x", padx=Sizing.PADDING_MEDIUM, pady=Sizing.PADDING_MEDIUM)

        self._widgets["processing_mode"] = ctk.CTkSegmentedButton(
            inner,
            values=list(self._mode_to_label.values()),
            command=self._on_mode_label_changed,
            selected_color=Colors.PRIMARY,
            selected_hover_color=Colors.PRIMARY_HOVER,
            unselected_color=Colors.BG_CARD,
            unselected_hover_color=Colors.BORDER_LIGHT,
            text_color=Colors.TEXT_PRIMARY,
        )
        self._widgets["processing_mode"].pack(fill="x")

        scan_row = ctk.CTkFrame(inner, fg_color="transparent")
        scan_row.pack(fill="x", pady=(Sizing.PADDING_MEDIUM, 0))
        scan_label = ctk.CTkLabel(
            scan_row,
            text=t("one_click_scan_interval"),
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
        )
        scan_label.pack(side="left")
        scan_tip = ctk.CTkLabel(
            scan_row,
            text="i",
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            cursor="hand2",
        )
        scan_tip.pack(side="left", padx=4)
        Tooltip(scan_tip, t("tip_one_click_scan_interval"))

        self._widgets["one_click_scan_interval"] = ValueOptionMenu(
            scan_row,
            options={
                "0.25": t("segments_scan_frequency_quarter"),
                "0.5": t("segments_scan_frequency_half"),
                "1.0": t("segments_scan_frequency_one"),
                "2.0": t("segments_scan_frequency_two"),
            },
            command=lambda _value: self._on_modified(),
            fg_color=Colors.BG_CARD,
            button_color=Colors.BG_CARD,
            button_hover_color=Colors.BORDER_LIGHT,
            dropdown_fg_color=Colors.BG_CARD,
            dropdown_hover_color=Colors.PRIMARY,
            text_color=Colors.TEXT_PRIMARY,
            width=190,
        )
        self._widgets["one_click_scan_interval"].pack(side="right")

        threshold_row = ctk.CTkFrame(inner, fg_color="transparent")
        threshold_row.pack(fill="x", pady=(Sizing.PADDING_SMALL, 0))
        threshold_label = ctk.CTkLabel(
            threshold_row,
            text=t("one_click_scan_threshold"),
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
        )
        threshold_label.pack(side="left")
        threshold_tip = ctk.CTkLabel(
            threshold_row,
            text="i",
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            cursor="hand2",
        )
        threshold_tip.pack(side="left", padx=4)
        Tooltip(threshold_tip, t("tip_one_click_scan_threshold"))
        self._widgets["one_click_scan_threshold_val"] = ctk.CTkLabel(
            threshold_row,
            text="0.70",
            text_color=Colors.TEXT_PRIMARY,
            width=38,
        )
        self._widgets["one_click_scan_threshold_val"].pack(side="right")
        self._widgets["one_click_scan_threshold"] = ctk.CTkSlider(
            threshold_row,
            from_=0.0,
            to=1.0,
            number_of_steps=20,
            fg_color=Colors.BG_CARD,
            progress_color=Colors.PRIMARY,
            button_color=Colors.PRIMARY,
            width=150,
            command=self._on_scan_threshold_changed,
        )
        self._widgets["one_click_scan_threshold"].pack(side="right", padx=(0, 8))
        self._widgets["one_click_scan_threshold"].set(0.70)

        confirmation_row = ctk.CTkFrame(inner, fg_color="transparent")
        confirmation_row.pack(fill="x", pady=(Sizing.PADDING_SMALL, 0))
        confirmation_label = ctk.CTkLabel(
            confirmation_row,
            text=t("one_click_confirmation"),
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_NORMAL),
        )
        confirmation_label.pack(side="left")
        confirmation_tip = ctk.CTkLabel(
            confirmation_row,
            text="i",
            text_color=Colors.TEXT_PRIMARY,
            font=(Fonts.FAMILY, Fonts.SIZE_TINY),
            cursor="hand2",
        )
        confirmation_tip.pack(side="left", padx=4)
        Tooltip(confirmation_tip, t("tip_one_click_confirmation"))
        self._widgets["one_click_min_consecutive_hits"] = ValueOptionMenu(
            confirmation_row,
            options={
                "1": t("one_click_confirmation_one"),
                "2": t("one_click_confirmation_two"),
                "3": t("one_click_confirmation_three"),
            },
            command=lambda _value: self._on_modified(),
            fg_color=Colors.BG_CARD,
            button_color=Colors.BG_CARD,
            button_hover_color=Colors.BORDER_LIGHT,
            dropdown_fg_color=Colors.BG_CARD,
            dropdown_hover_color=Colors.PRIMARY,
            text_color=Colors.TEXT_PRIMARY,
            width=150,
        )
        self._widgets["one_click_min_consecutive_hits"].pack(side="right")
        self._widgets["one_click_min_consecutive_hits"].set_value("2")
        self._set_scan_state(PROCESSING_MODE_STANDARD)

    def _on_scan_threshold_changed(self, value: float) -> None:
        self._widgets["one_click_scan_threshold_val"].configure(
            text=f"{value:.2f}"
        )
        self._on_modified()

    def _on_mode_label_changed(self, label: str) -> None:
        mode = self._label_to_mode[label]
        self._set_scan_state(mode)
        self._on_modified()

    def _set_scan_state(self, mode: str) -> None:
        state = "normal" if mode == PROCESSING_MODE_ONE_CLICK_VR else "disabled"
        for key in (
            "one_click_scan_interval",
            "one_click_scan_threshold",
            "one_click_min_consecutive_hits",
        ):
            self._widgets[key].configure(state=state)

    def set_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self._widgets["processing_mode"].configure(state=state)
        if not enabled:
            for key in (
                "one_click_scan_interval",
                "one_click_scan_threshold",
                "one_click_min_consecutive_hits",
            ):
                self._widgets[key].configure(state="disabled")
            return
        label = self._widgets["processing_mode"].get()
        self._set_scan_state(self._label_to_mode[label])

    def apply(self, preset) -> None:
        mode = str(preset.processing_mode)
        if mode not in self._mode_to_label:
            mode = PROCESSING_MODE_STANDARD
        self._widgets["processing_mode"].set(self._mode_to_label[mode])
        interval = str(float(preset.one_click_scan_interval))
        self._widgets["one_click_scan_interval"].set_value(interval)
        self._widgets["one_click_scan_threshold"].set(
            float(preset.one_click_scan_threshold)
        )
        self._widgets["one_click_scan_threshold_val"].configure(
            text=f"{float(preset.one_click_scan_threshold):.2f}"
        )
        self._widgets["one_click_min_consecutive_hits"].set_value(
            str(int(preset.one_click_min_consecutive_hits))
        )
        self._set_scan_state(mode)

    def collect(self) -> dict:
        label = self._widgets["processing_mode"].get()
        return {
            "processing_mode": self._label_to_mode[label],
            "one_click_scan_interval": float(
                self._widgets["one_click_scan_interval"].get_value()
            ),
            "one_click_scan_threshold": float(
                self._widgets["one_click_scan_threshold"].get()
            ),
            "one_click_min_consecutive_hits": int(
                self._widgets["one_click_min_consecutive_hits"].get_value()
            ),
        }
