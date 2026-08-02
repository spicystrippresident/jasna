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
        self._set_scan_state(PROCESSING_MODE_STANDARD)

    def _on_mode_label_changed(self, label: str) -> None:
        mode = self._label_to_mode[label]
        self._set_scan_state(mode)
        self._on_modified()

    def _set_scan_state(self, mode: str) -> None:
        state = "normal" if mode == PROCESSING_MODE_ONE_CLICK_VR else "disabled"
        self._widgets["one_click_scan_interval"].configure(state=state)

    def set_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self._widgets["processing_mode"].configure(state=state)
        if not enabled:
            self._widgets["one_click_scan_interval"].configure(state="disabled")
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
        self._set_scan_state(mode)

    def collect(self) -> dict:
        label = self._widgets["processing_mode"].get()
        return {
            "processing_mode": self._label_to_mode[label],
            "one_click_scan_interval": float(
                self._widgets["one_click_scan_interval"].get_value()
            ),
        }
