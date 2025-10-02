(* C用：補正後LCBの式カードを保存 *)
m = 3; z = 1.96;
rule = "LCB95_adj = tau - " <> ToString[z] <> "*sqrt(m)*se";
Export["reports/figures/selection_rule.txt", rule, "Text"];
