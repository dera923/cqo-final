(* Assumptions *)
$Assumptions = 0 < x < 1;
(* A用：Jカーブの可視化（x(1-x)） *)
Export["reports/figures/j_curve.png", Plot[x (1 - x), {x, 0, 1}], "PNG"];
