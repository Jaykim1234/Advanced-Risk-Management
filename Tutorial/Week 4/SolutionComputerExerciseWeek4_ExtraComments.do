cls
clear all

cd "C:\Users\Jinhyun\Documents\GitHub\Advanced-Risk-Management\Tutorial\Week 4"  // change to your working directory / folder
// Question 1
use  StockIndex.dta
bcal load index 
tsline Dax30, name(Dax)
tsline SP500, name(SP500)
tsline FTSE100, name(FTSE)
tsline Nikkei225, name(Nikkei)
graph combine Dax SP500 FTSE Nikkei, name(Returns)
graph close Dax SP500 FTSE Nikkei 
// (1) Peaks around crisis (1991,2000,2008);
// (2) Clearly similar patterns in volatility, hard to see return correlations
//	   (requires information about same or opposite sign on the same day, scatter diagram)

// Q2
mgarch ccc (SP500 Dax30), arch(1) garch(1)
predict v_ccc*, variance // output is three variables: three combinations between two stocks (SPX/SPX, SPX/DAX, DAX,DAX)
predict r_ccc*, correlation // output is three variables (as for the variance)
estimates store CCC

// Additional:
// Self-check for correlation matrix value
g condcorr_SP500_Dax30 = v_ccc_Dax30_SP500/(sqrt(v_ccc_SP500_SP500)*sqrt(v_ccc_Dax30_Dax30))

//Q3
mgarch dcc (SP500 Dax30), arch(1) garch(1)
predict v_dcc*, variance
predict r_dcc*, correlation
estimates store DCC

// Obtain parameter estimates: Adjustment parameters 
g alpha=_b[Adjustment:lambda1] // extract first coefficient
g beta = _b[Adjustment:lambda2] // extract second coefficient
g alpha_plus_beta = alpha+beta // obtain sum of coefficients (check whether it is less than 1)


//Q4 
gen rho_CCC = r_ccc_Dax30_SP500
gen rho_DCC = r_dcc_Dax30_SP500 
tsline  rho_CCC rho_DCC, name(CCC_vs_DCC)
// (1) Constant versus time-varying correlations;
// (2) Clear peaks around crisis and time varying patterns 

lrtest CCC DCC
// Reject the null 

//Q5
gen Vol_p_CCC= sqrt(v_ccc_SP500_SP500/4 +v_ccc_Dax30_Dax30/4+ v_ccc_Dax30_SP500/2)
gen Vol_p_DCC= sqrt(v_dcc_SP500_SP500/4 +v_dcc_Dax30_Dax30/4+ v_dcc_Dax30_SP500/2)
tsline Vol_p_CCC Vol_p_DCC, name(Portf_Vol)

// Differences are very small; both have peaks around crisis;
// DCC leads to higher peaks (reasonable result compared with the Q4 figure);
// CCC leads to lower volatility in the 1990s

//Q6 
mgarch dcc (SP500 Dax30 FTSE100 Nikkei225), arch(1) garch(1)
predict r_dcc4*, correlation
gen rho_DCC4 = r_dcc4_Dax30_SP500
tsline rho_DCC rho_DCC4, name(DCC_2_vs_4)
// Quite similar patterns though not identical. 
