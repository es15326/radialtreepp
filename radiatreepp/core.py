import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Wedge
from matplotlib import cm
import matplotlib

from typing import Optional, Dict
from .config import RadialTreeConfig  # import your config dataclass


def radialTreee(Z2, ax=None, config: RadialTreeConfig = RadialTreeConfig()):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    R = 1.0
    if not Z2['icoord']:
        ax.axis('off')
        return ax
        
    xmax = np.max(Z2['icoord'])
    ymax = np.max(Z2['dcoord'])

    gradient_colors = LinearSegmentedColormap.from_list("custom_gradient", config.gradient_colors)
    normalize = lambda d: (d - config.gradient_min) / (config.gradient_max - config.gradient_min) if (config.gradient_max - config.gradient_min) != 0 else 0
    color_map = lambda d: gradient_colors(normalize(d))

    # --- Pre-calculation for 'top_3' label mode ---
    num_merges = len(Z2['dcoord'])
    top_3_label_indices = []
    if config.node_label_display_mode == 'top_3':
        # The last 3 merges in the list are the highest ones
        top_3_label_indices = range(num_merges - 3, num_merges)

    for i, (icoord, dcoord, _) in enumerate(zip(Z2['icoord'], Z2['dcoord'], Z2['color_list'])):
        avg_depth = np.mean(dcoord)
        c = color_map(avg_depth)
        r = R * (1 - np.array(dcoord) / ymax) if ymax > 0 else np.ones_like(dcoord) * R
        
        theta_branches = 2 * np.pi * np.array([icoord[0], icoord[2]]) / xmax

        x0, y0 = r[0] * np.cos(theta_branches[0]), r[0] * np.sin(theta_branches[0])
        x1, y1 = r[1] * np.cos(theta_branches[0]), r[1] * np.sin(theta_branches[0])
        x2, y2 = r[2] * np.cos(theta_branches[1]), r[2] * np.sin(theta_branches[1])
        x3, y3 = r[3] * np.cos(theta_branches[1]), r[3] * np.sin(theta_branches[1])

        ax.plot([x0, x1], [y0, y1], color=c, linewidth=config.line_width)
        ax.plot([x2, x3], [y2, y3], color=c, linewidth=config.line_width)
        
        arc_radius = r[1]
        t0, t1 = min(theta_branches), max(theta_branches)
        if t1 - t0 > np.pi:
            t0, t1 = t1, t0 + 2*np.pi
            
        arc_theta = np.linspace(t0, t1, 100)
        arc_x = arc_radius * np.cos(arc_theta)
        arc_y = arc_radius * np.sin(arc_theta)
        ax.plot(arc_x, arc_y, color=c, linewidth=config.line_width)

        # --- Updated logic for drawing nodes and labels ---
        # An "inner" node connects two previous clusters (not original leaves)
        is_inner_node = dcoord[0] > 0 and dcoord[3] > 0
        
        # Check if we should draw the node based on the display mode
        should_draw_node = (config.node_display_mode == 'all') or \
                           (config.node_display_mode == 'inner' and is_inner_node)

        # Check if we should draw the label based on the display mode
        should_draw_label = (config.node_label_display_mode == 'all') or \
                            (config.node_label_display_mode == 'inner' and is_inner_node) or \
                            (config.node_label_display_mode == 'top_3' and i in top_3_label_indices)

        if should_draw_node or should_draw_label:
            mid_angle = (t0 + t1) / 2.0
            
            if should_draw_node:
                node_x = arc_radius * np.cos(mid_angle)
                node_y = arc_radius * np.sin(mid_angle)
                ax.plot(node_x, node_y, marker='o', color=config.node_color, 
                        markersize=config.node_size, linestyle='none')

            if should_draw_label:
                label_text = f"{dcoord[1]:.2f}"
                label_radius_offset = arc_radius + 0.035
                label_x = label_radius_offset * np.cos(mid_angle)
                label_y = label_radius_offset * np.sin(mid_angle)
                rotation = np.rad2deg(mid_angle)
                if 90 < rotation < 270:
                    rotation += 180
                ax.text(label_x, label_y, label_text, ha='center', va='center',
                        fontsize=config.node_label_fontsize, rotation=rotation,
                        rotation_mode='anchor')

    if config.addlabels:
        for i, label in enumerate(Z2['ivl']):
            angle = (5 + i * 10.0) / xmax * 2 * np.pi
            x = config.label_radius * np.cos(angle)
            y = config.label_radius * np.sin(angle)
            
            if config.radial_labels:
                ha = 'left' if x >= 0 else 'right'
                rotation = np.rad2deg(angle)
                if x < 0: rotation += 180
                ax.text(x, y, label, ha=ha, va='center', fontsize=config.fontsize, rotation=rotation, rotation_mode='anchor')
            else:
                ha = 'center'
                if x > 0.1: ha = 'left'
                elif x < -0.1: ha = 'right'
                va = 'center'
                if y > 0.1: va = 'bottom'
                elif y < -0.1: va = 'top'
                ax.text(x, y, label, ha=ha, va=va, fontsize=config.fontsize, rotation=0)

    if config.colorlabels:
        for j, (ring_name, color_list) in enumerate(config.colorlabels.items()):
            radius = R + 0.05 + j * (config.ring_width + config.ring_spacing)
            theta_start = (5.0 / xmax) * 2 * np.pi
            values = np.ones(len(Z2['ivl']))
            ax.pie(values, colors=np.array(color_list)[Z2['leaves']], radius=radius,
                   counterclock=True, startangle=np.rad2deg(theta_start),
                   wedgeprops=dict(width=config.ring_width))

    ax.set_aspect('equal')
    ax.axis('off')
    max_pie_radius = R + 0.05 + (len(config.colorlabels) if config.colorlabels else 0) * (config.ring_width + config.ring_spacing)
    plot_limit = max(max_pie_radius, config.label_radius) + 0.2
    ax.set_xlim(-plot_limit, plot_limit)
    ax.set_ylim(-plot_limit, plot_limit)
    return ax
