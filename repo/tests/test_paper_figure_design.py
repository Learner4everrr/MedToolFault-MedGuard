"""Presentation-level invariants; no experiment data or model inference needed."""
import importlib.util
from pathlib import Path
import unittest

import matplotlib.pyplot as plt
import numpy as np

P = Path(__file__).resolve().parents[1] / "paper_nc_v1" / "figure_design.py"
spec = importlib.util.spec_from_file_location("paper_figure_design", P)
design = importlib.util.module_from_spec(spec)
spec.loader.exec_module(design)


class FigureDesignTests(unittest.TestCase):
    def test_times_typography_has_real_font_files_for_all_styles(self):
        with plt.rc_context():
            design.configure()
            family = plt.rcParams["font.family"][0]
            self.assertIn(family, design.TIMES_FAMILIES)
            self.assertEqual(plt.rcParams["mathtext.rm"], family)
            for style, weight in (("normal", "normal"), ("normal", "bold"),
                                  ("italic", "normal"), ("italic", "bold")):
                props = design.font_manager.FontProperties(family=family, style=style, weight=weight)
                path = design.font_manager.findfont(props, fallback_to_default=False)
                self.assertTrue(Path(path).is_file())
                self.assertEqual(design.font_manager.FontProperties(fname=path).get_name(), family)

    def tearDown(self):
        plt.close("all")

    def test_canvas_has_fixed_publication_width(self):
        fig = design.canvas(169)
        self.assertAlmostEqual(fig.get_figwidth()*25.4,183)
        self.assertAlmostEqual(fig.get_figheight()*25.4,169)

    def test_comparison_reversal_preserves_interval_order(self):
        r = dict(difference=.2,ci95=[.1,.3])
        self.assertEqual(design.effect(r,True),(-20,-30,-10))

    def test_primary_effect_uses_frozen_estimand(self):
        r = dict(task_weighted_gain=.2,patient_cluster_ci95=[.1,.3])
        self.assertEqual(design.effect(r),(20,10,30))

    def test_interval_clipping_is_rejected(self):
        fig,ax=plt.subplots()
        design.forest(ax,["Model"],(-10,10))
        with self.assertRaisesRegex(AssertionError,"clip an interval"):
            design.dot(ax,0,dict(difference=.05,ci95=[-.02,.2]))

    def test_heatmap_clipping_is_rejected(self):
        fig,ax=plt.subplots()
        with self.assertRaisesRegex(AssertionError,"clip data"):
            design.heat(ax,[[110]],["Model"],["Policy"],bound=100)

    def test_signed_scale_is_symmetric(self):
        fig,ax=plt.subplots()
        im=design.heat(ax,[[-8,12]],["Model"],["A","B"],bound=25,signed=True)
        self.assertEqual((im.norm.vmin,im.norm.vmax),(-25,25))

    def test_heatmap_text_has_adequate_contrast(self):
        for cm in (design.CHANGE,design.SUCCESS,design.HARM,design.COST):
            for value in np.linspace(0,1,256):
                rgb=cm(value); fg=design.contrast_text(rgb)
                lum=design.luminance(rgb)
                ratio=(lum+.05)/.05 if fg=="#000000" else 1.05/(lum+.05)
                self.assertGreaterEqual(ratio,4.5)

    def test_zero_markers_are_not_on_clipped_axis_edge(self):
        fig,ax=plt.subplots()
        design.policy_points(ax,[[0,20]],["Model"],("vanilla","medguard"),60)
        self.assertLess(ax.get_xlim()[0],0)
        self.assertGreater(ax.get_xlim()[1],60)


if __name__ == "__main__":
    unittest.main()
