(* DR/EIF カード：式の整形と保存 *)
$Assumptions = 0<e<1 && Element[{Y,mu0,mu1},Reals] && Element[T,Integers] && (T==0||T==1);
expr = (mu1 - mu0) + T/e*(Y - mu1) - (1 - T)/(1 - e)*(Y - mu0);
Export["reports/figures/dr_eif_card.txt", ToString[expr, InputForm], "Text"];
