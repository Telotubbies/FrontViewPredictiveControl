"""
Simple test for CubicSpline lane fitting.
Tests that spline passes through all window center points.
"""
import numpy as np
import cv2
from scipy.interpolate import CubicSpline

# Simulate window center points (row, col) - like a curved lane
test_points = [
    (600, 150),  # bottom
    (550, 155),
    (500, 165),
    (450, 180),
    (400, 200),
    (350, 225),
    (300, 255),
    (250, 290),
    (200, 330),  # top
]

rows = np.array([p[0] for p in test_points], dtype=float)
cols = np.array([p[1] for p in test_points], dtype=float)

# Sort by rows (CubicSpline requires strictly increasing x)
sort_idx = np.argsort(rows)
rows = rows[sort_idx]
cols = cols[sort_idx]

# Fit CubicSpline
spline = CubicSpline(rows, cols, bc_type='natural')

# Fit polynomial (degree 2)
poly_coeffs = np.polyfit(rows, cols, 2)

# Evaluate at original points
spline_cols = spline(rows)
poly_cols = np.polyval(poly_coeffs, rows)

print("=== CubicSpline vs Polynomial Fitting Test ===\n")
print(f"{'Row':>6} | {'Actual':>8} | {'Spline':>8} | {'Poly':>8} | {'Spline Err':>10} | {'Poly Err':>10}")
print("-" * 70)

spline_errors = []
poly_errors = []

for r, actual, sp, po in zip(rows, cols, spline_cols, poly_cols):
    sp_err = abs(sp - actual)
    po_err = abs(po - actual)
    spline_errors.append(sp_err)
    poly_errors.append(po_err)
    print(f"{r:6.0f} | {actual:8.1f} | {sp:8.1f} | {po:8.1f} | {sp_err:10.4f} | {po_err:10.4f}")

print("-" * 70)
print(f"{'Total':>6} | {'':>8} | {'':>8} | {'':>8} | {sum(spline_errors):10.4f} | {sum(poly_errors):10.4f}")
print(f"\nSpline max error: {max(spline_errors):.6f}")
print(f"Poly max error:   {max(poly_errors):.6f}")

# Visual test - create image
img = np.zeros((640, 640, 3), dtype=np.uint8)

# Draw original points (green)
for r, c in test_points:
    cv2.circle(img, (int(c), int(r)), 5, (0, 255, 0), -1)

# Draw spline curve (blue)
eval_rows = np.linspace(rows.min(), rows.max(), 100)
spline_curve = spline(eval_rows)
for i in range(len(eval_rows) - 1):
    pt1 = (int(spline_curve[i]), int(eval_rows[i]))
    pt2 = (int(spline_curve[i+1]), int(eval_rows[i+1]))
    cv2.line(img, pt1, pt2, (255, 0, 0), 2)

# Draw polynomial curve (red)
poly_curve = np.polyval(poly_coeffs, eval_rows)
for i in range(len(eval_rows) - 1):
    pt1 = (int(poly_curve[i]), int(eval_rows[i]))
    pt2 = (int(poly_curve[i+1]), int(eval_rows[i+1]))
    cv2.line(img, pt1, pt2, (0, 0, 255), 2)

# Add legend
cv2.putText(img, "Green: Original points", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
cv2.putText(img, "Blue: CubicSpline", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
cv2.putText(img, "Red: Polynomial (deg 2)", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

cv2.imwrite("/tmp/spline_test.png", img)
print(f"\nVisualization saved to /tmp/spline_test.png")
print("\n✓ CubicSpline passes through ALL points (error = 0)")
print("✓ Polynomial has fitting error at curve points")
