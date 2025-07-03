from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass
class RadialTreeConfig:
    fontsize: int = 8
    label_radius: float = 1.2
    addlabels: bool = True
    radial_labels: bool = False
    line_width: float = 0.5
    ring_width: float = 0.05
    ring_spacing: float = 0.02
    gradient_min: float = 0.0
    gradient_max: float = 4.0
    gradient_colors: list = field(default_factory=lambda: ["black", "blue"])
    colorlabels: Optional[Dict] = None
    node_display_mode: str = 'inner'
    node_label_display_mode: str = 'top_3'
    node_size: int = 4
    node_color: str = 'black'
    node_label_fontsize: int = 6
