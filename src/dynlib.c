# include "dynlib.h"
# include <math.h>
# include <stdio.h>

double thetadd(double *y, double La, double Ls, double ds, double mb, double rc, double Ia, double mp, double g, double T_drive){
 double theta, thetad, psi, psid;
 theta = y[0];
 thetad = y[1];
 psi = y[2];
 psid = y[3];
 return ((-0.25*pow(La,2)*pow(Ls,2)*pow(ds,2)*pow(thetad,2)*sin(psi)*cos(psi) - 1.0*pow(La,2)*Ls*ds*mp*pow(thetad,2)*sin(psi)*cos(psi) - 1.0*pow(La,2)*pow(mp,2)*pow(thetad,2)*sin(psi)*cos(psi) + 0.166666666666667*La*pow(Ls,3)*pow(ds,2)*pow(psid,2)*sin(psi) - 0.333333333333333*La*pow(Ls,3)*pow(ds,2)*psid*thetad*sin(psi) + 0.166666666666667*La*pow(Ls,3)*pow(ds,2)*pow(thetad,2)*sin(psi) + 0.25*La*pow(Ls,2)*pow(ds,2)*g*sin(psi - theta)*cos(psi) + 0.333333333333333*La*pow(Ls,2)*pow(ds,2)*g*sin(theta) + 0.833333333333333*La*pow(Ls,2)*ds*mp*pow(psid,2)*sin(psi) - 1.66666666666667*La*pow(Ls,2)*ds*mp*psid*thetad*sin(psi) + 0.833333333333333*La*pow(Ls,2)*ds*mp*pow(thetad,2)*sin(psi) + 1.0*La*Ls*ds*g*mp*pow(sin(psi),2)*pow(sin(theta),3) + 0.25*La*Ls*ds*g*mp*pow(sin(psi),2)*sin(theta) + 0.25*La*Ls*ds*g*mp*sin(psi)*cos(psi - 3*theta) - 1.0*La*Ls*ds*g*mp*sin(psi)*cos(psi)*pow(cos(theta),3) + 1.75*La*Ls*ds*g*mp*sin(psi)*cos(psi)*cos(theta) - 0.5*La*Ls*ds*g*mp*pow(sin(theta),3) + 0.708333333333333*La*Ls*ds*g*mp*sin(theta) - 0.125*La*Ls*ds*g*mp*sin(3*theta) + 1.0*La*Ls*pow(mp,2)*pow(psid,2)*sin(psi) - 2.0*La*Ls*pow(mp,2)*psid*thetad*sin(psi) + 1.0*La*Ls*pow(mp,2)*pow(thetad,2)*sin(psi) + 2.0*La*g*pow(mp,2)*pow(sin(psi),2)*pow(sin(theta),3) - 0.5*La*g*pow(mp,2)*pow(sin(psi),2)*sin(theta) + 0.5*La*g*pow(mp,2)*sin(psi)*cos(psi - 3*theta) - 2.0*La*g*pow(mp,2)*sin(psi)*cos(psi)*pow(cos(theta),3) + 2.5*La*g*pow(mp,2)*sin(psi)*cos(psi)*cos(theta) - 1.0*La*g*pow(mp,2)*pow(sin(theta),3) + 0.75*La*g*pow(mp,2)*sin(theta) - 0.25*La*g*pow(mp,2)*sin(3*theta) - 0.333333333333333*Ls*T_drive*ds + 0.333333333333333*Ls*ds*g*mb*rc*sin(theta) - 1.0*T_drive*mp + 1.0*g*mb*mp*rc*sin(theta))/(0.333333333333333*Ia*Ls*ds + 1.0*Ia*mp + 0.25*pow(La,2)*pow(Ls,2)*pow(ds,2)*pow(sin(psi),2) + 0.0833333333333333*pow(La,2)*pow(Ls,2)*pow(ds,2) + 1.0*pow(La,2)*Ls*ds*mp*pow(sin(psi),2) + 0.333333333333333*pow(La,2)*Ls*ds*mp + 1.0*pow(La,2)*pow(mp,2)*pow(sin(psi),2)));
}

double psidd(double *y, double La, double Ls, double ds, double mb, double rc, double Ia, double mp, double g, double T_drive){
 double theta, thetad, psi, psid;
 theta = y[0];
 thetad = y[1];
 psi = y[2];
 psid = y[3];
 return ((0.166666666666667*Ia*La*pow(Ls,2)*pow(ds,2)*pow(thetad,2)*sin(psi) + 0.833333333333333*Ia*La*Ls*ds*mp*pow(thetad,2)*sin(psi) + 1.0*Ia*La*pow(mp,2)*pow(thetad,2)*sin(psi) - 0.166666666666667*Ia*pow(Ls,2)*pow(ds,2)*g*sin(psi - theta) - 0.833333333333333*Ia*Ls*ds*g*mp*sin(psi - theta) - 1.0*Ia*g*pow(mp,2)*sin(psi - theta) + 0.166666666666667*pow(La,3)*pow(Ls,3)*pow(ds,3)*pow(thetad,2)*sin(psi) + 1.0*pow(La,3)*pow(Ls,2)*pow(ds,2)*mp*pow(thetad,2)*sin(psi) + 1.83333333333333*pow(La,3)*Ls*ds*pow(mp,2)*pow(thetad,2)*sin(psi) + 1.0*pow(La,3)*pow(mp,3)*pow(thetad,2)*sin(psi) - 0.0833333333333333*pow(La,2)*pow(Ls,4)*pow(ds,3)*pow(psid,2)*sin(psi)*cos(psi) + 0.166666666666667*pow(La,2)*pow(Ls,4)*pow(ds,3)*psid*thetad*sin(psi)*cos(psi) - 0.166666666666667*pow(La,2)*pow(Ls,4)*pow(ds,3)*pow(thetad,2)*sin(psi)*cos(psi) - 0.166666666666667*pow(La,2)*pow(Ls,3)*pow(ds,3)*g*sin(psi)*cos(theta) - 0.583333333333333*pow(La,2)*pow(Ls,3)*pow(ds,2)*mp*pow(psid,2)*sin(psi)*cos(psi) + 1.16666666666667*pow(La,2)*pow(Ls,3)*pow(ds,2)*mp*psid*thetad*sin(psi)*cos(psi) - 1.16666666666667*pow(La,2)*pow(Ls,3)*pow(ds,2)*mp*pow(thetad,2)*sin(psi)*cos(psi) + 0.0520833333333333*pow(La,2)*pow(Ls,2)*pow(ds,2)*g*mp*sin(psi - 3*theta) - 0.03125*pow(La,2)*pow(Ls,2)*pow(ds,2)*g*mp*sin(3*psi - 3*theta) - 0.75*pow(La,2)*pow(Ls,2)*pow(ds,2)*g*mp*pow(sin(psi),3)*pow(cos(theta),3) + 0.375*pow(La,2)*pow(Ls,2)*pow(ds,2)*g*mp*pow(sin(psi),3)*cos(theta) + 0.25*pow(La,2)*pow(Ls,2)*pow(ds,2)*g*mp*sin(psi)*pow(sin(theta),2)*cos(psi - theta)*cos(psi) - 0.25*pow(La,2)*pow(Ls,2)*pow(ds,2)*g*mp*sin(psi)*cos(psi - theta)*cos(psi) + 0.416666666666667*pow(La,2)*pow(Ls,2)*pow(ds,2)*g*mp*sin(psi)*pow(cos(theta),3) - 1.125*pow(La,2)*pow(Ls,2)*pow(ds,2)*g*mp*sin(psi)*cos(theta) + 0.75*pow(La,2)*pow(Ls,2)*pow(ds,2)*g*mp*pow(sin(theta),3)*pow(cos(psi),3) - 0.833333333333333*pow(La,2)*pow(Ls,2)*pow(ds,2)*g*mp*pow(sin(theta),3)*cos(psi) - 0.625*pow(La,2)*pow(Ls,2)*pow(ds,2)*g*mp*sin(theta)*pow(cos(psi),3) + 0.6875*pow(La,2)*pow(Ls,2)*pow(ds,2)*g*mp*sin(theta)*cos(psi) - 1.33333333333333*pow(La,2)*pow(Ls,2)*ds*pow(mp,2)*pow(psid,2)*sin(psi)*cos(psi) + 2.66666666666667*pow(La,2)*pow(Ls,2)*ds*pow(mp,2)*psid*thetad*sin(psi)*cos(psi) - 2.66666666666667*pow(La,2)*pow(Ls,2)*ds*pow(mp,2)*pow(thetad,2)*sin(psi)*cos(psi) - 1.0*pow(La,2)*Ls*ds*g*pow(mp,2)*sin(psi - theta)*sin(psi)*sin(theta)*cos(psi)*pow(cos(theta),3) + 0.25*pow(La,2)*Ls*ds*g*pow(mp,2)*sin(psi + theta)*pow(sin(theta),4) + 0.046875*pow(La,2)*Ls*ds*g*pow(mp,2)*sin(3*psi - 3*theta) + 0.25*pow(La,2)*Ls*ds*g*pow(mp,2)*sin(3*psi - theta)*pow(sin(theta),4) - 0.5*pow(La,2)*Ls*ds*g*pow(mp,2)*sin(3*psi - theta)*pow(sin(theta),2) + 0.75*pow(La,2)*Ls*ds*g*pow(mp,2)*pow(sin(psi),3)*pow(cos(theta),3) - 0.25*pow(La,2)*Ls*ds*g*pow(mp,2)*pow(sin(psi),3)*cos(theta) - 0.25*pow(La,2)*Ls*ds*g*pow(mp,2)*sin(psi)*cos(psi - 3*theta)*cos(psi) - 0.958333333333333*pow(La,2)*Ls*ds*g*pow(mp,2)*sin(psi)*pow(cos(theta),3) - 1.41145833333333*pow(La,2)*Ls*ds*g*pow(mp,2)*sin(psi)*cos(theta) + 0.223958333333333*pow(La,2)*Ls*ds*g*pow(mp,2)*sin(psi)*cos(3*theta) + 0.140625*pow(La,2)*Ls*ds*g*pow(mp,2)*sin(3*psi)*cos(theta) + 0.25*pow(La,2)*Ls*ds*g*pow(mp,2)*pow(sin(theta),3)*pow(cos(psi),3) - 0.458333333333333*pow(La,2)*Ls*ds*g*pow(mp,2)*pow(sin(theta),3)*cos(psi) + 0.25*pow(La,2)*Ls*ds*g*pow(mp,2)*sin(theta)*pow(cos(psi),3) - 0.109375*pow(La,2)*Ls*ds*g*pow(mp,2)*sin(theta)*cos(psi) - 0.359375*pow(La,2)*Ls*ds*g*pow(mp,2)*sin(theta)*cos(3*psi) + 0.119791666666667*pow(La,2)*Ls*ds*g*pow(mp,2)*sin(3*theta)*cos(psi) - 1.0*pow(La,2)*Ls*pow(mp,3)*pow(psid,2)*sin(psi)*cos(psi) + 2.0*pow(La,2)*Ls*pow(mp,3)*psid*thetad*sin(psi)*cos(psi) - 2.0*pow(La,2)*Ls*pow(mp,3)*pow(thetad,2)*sin(psi)*cos(psi) - 2.0*pow(La,2)*g*pow(mp,3)*sin(psi - theta)*sin(psi)*sin(theta)*cos(psi)*pow(cos(theta),3) + 0.5*pow(La,2)*g*pow(mp,3)*sin(psi + theta)*pow(sin(theta),4) + 0.15625*pow(La,2)*g*pow(mp,3)*sin(3*psi - 3*theta) + 0.5*pow(La,2)*g*pow(mp,3)*sin(3*psi - theta)*pow(sin(theta),4) - 1.0*pow(La,2)*g*pow(mp,3)*sin(3*psi - theta)*pow(sin(theta),2) + 2.5*pow(La,2)*g*pow(mp,3)*pow(sin(psi),3)*pow(cos(theta),3) - 1.0*pow(La,2)*g*pow(mp,3)*pow(sin(psi),3)*cos(theta) - 0.5*pow(La,2)*g*pow(mp,3)*sin(psi)*cos(psi - 3*theta)*cos(psi) - 2.25*pow(La,2)*g*pow(mp,3)*sin(psi)*pow(cos(theta),3) - 0.09375*pow(La,2)*g*pow(mp,3)*sin(psi)*cos(theta) + 0.34375*pow(La,2)*g*pow(mp,3)*sin(psi)*cos(3*theta) + 0.34375*pow(La,2)*g*pow(mp,3)*sin(3*psi)*cos(theta) - 0.5*pow(La,2)*g*pow(mp,3)*pow(sin(theta),3)*pow(cos(psi),3) - 0.25*pow(La,2)*g*pow(mp,3)*pow(sin(theta),3)*cos(psi) + 1.0*pow(La,2)*g*pow(mp,3)*sin(theta)*pow(cos(psi),3) - 0.53125*pow(La,2)*g*pow(mp,3)*sin(theta)*cos(psi) - 0.65625*pow(La,2)*g*pow(mp,3)*sin(theta)*cos(3*psi) + 0.21875*pow(La,2)*g*pow(mp,3)*sin(3*theta)*cos(psi) + 0.0555555555555556*La*pow(Ls,5)*pow(ds,3)*pow(psid,2)*sin(psi) - 0.111111111111111*La*pow(Ls,5)*pow(ds,3)*psid*thetad*sin(psi) + 0.0555555555555556*La*pow(Ls,5)*pow(ds,3)*pow(thetad,2)*sin(psi) + 0.0833333333333333*La*pow(Ls,4)*pow(ds,3)*g*sin(psi - theta)*cos(psi) + 0.111111111111111*La*pow(Ls,4)*pow(ds,3)*g*sin(theta) + 0.444444444444444*La*pow(Ls,4)*pow(ds,2)*mp*pow(psid,2)*sin(psi) - 0.888888888888889*La*pow(Ls,4)*pow(ds,2)*mp*psid*thetad*sin(psi) + 0.444444444444444*La*pow(Ls,4)*pow(ds,2)*mp*pow(thetad,2)*sin(psi) + 0.333333333333333*La*pow(Ls,3)*pow(ds,2)*g*mp*pow(sin(psi),2)*pow(sin(theta),3) + 0.333333333333333*La*pow(Ls,3)*pow(ds,2)*g*mp*pow(sin(psi),2)*sin(theta) + 0.0833333333333333*La*pow(Ls,3)*pow(ds,2)*g*mp*sin(psi)*cos(psi - 3*theta) - 0.333333333333333*La*pow(Ls,3)*pow(ds,2)*g*mp*sin(psi)*cos(psi)*pow(cos(theta),3) + 0.833333333333333*La*pow(Ls,3)*pow(ds,2)*g*mp*sin(psi)*cos(psi)*cos(theta) - 0.166666666666667*La*pow(Ls,3)*pow(ds,2)*g*mp*pow(sin(theta),3) + 0.319444444444444*La*pow(Ls,3)*pow(ds,2)*g*mp*sin(theta) - 0.0416666666666667*La*pow(Ls,3)*pow(ds,2)*g*mp*sin(3*theta) + 1.16666666666667*La*pow(Ls,3)*ds*pow(mp,2)*pow(psid,2)*sin(psi) - 2.33333333333333*La*pow(Ls,3)*ds*pow(mp,2)*psid*thetad*sin(psi) + 1.16666666666667*La*pow(Ls,3)*ds*pow(mp,2)*pow(thetad,2)*sin(psi) + 0.166666666666667*La*pow(Ls,2)*T_drive*pow(ds,2)*cos(psi) - 0.166666666666667*La*pow(Ls,2)*pow(ds,2)*g*mb*rc*sin(theta)*cos(psi) + 1.66666666666667*La*pow(Ls,2)*ds*g*pow(mp,2)*pow(sin(psi),2)*pow(sin(theta),3) + 0.0833333333333333*La*pow(Ls,2)*ds*g*pow(mp,2)*pow(sin(psi),2)*sin(theta) + 0.416666666666667*La*pow(Ls,2)*ds*g*pow(mp,2)*sin(psi)*cos(psi - 3*theta) - 1.66666666666667*La*pow(Ls,2)*ds*g*pow(mp,2)*sin(psi)*cos(psi)*pow(cos(theta),3) + 2.58333333333333*La*pow(Ls,2)*ds*g*pow(mp,2)*sin(psi)*cos(psi)*cos(theta) - 0.833333333333333*La*pow(Ls,2)*ds*g*pow(mp,2)*pow(sin(theta),3) + 0.958333333333333*La*pow(Ls,2)*ds*g*pow(mp,2)*sin(theta) - 0.208333333333333*La*pow(Ls,2)*ds*g*pow(mp,2)*sin(3*theta) + 1.0*La*pow(Ls,2)*pow(mp,3)*pow(psid,2)*sin(psi) - 2.0*La*pow(Ls,2)*pow(mp,3)*psid*thetad*sin(psi) + 1.0*La*pow(Ls,2)*pow(mp,3)*pow(thetad,2)*sin(psi) + 0.833333333333333*La*Ls*T_drive*ds*mp*cos(psi) + 0.0833333333333333*La*Ls*ds*g*mb*mp*rc*sin(psi - 3*theta) + 0.333333333333333*La*Ls*ds*g*mb*mp*rc*sin(psi - theta)*pow(sin(theta),2) - 0.0833333333333333*La*Ls*ds*g*mb*mp*rc*sin(psi)*cos(theta) - 0.583333333333333*La*Ls*ds*g*mb*mp*rc*sin(theta)*cos(psi) + 2.0*La*Ls*g*pow(mp,3)*pow(sin(psi),2)*pow(sin(theta),3) - 0.5*La*Ls*g*pow(mp,3)*pow(sin(psi),2)*sin(theta) + 0.5*La*Ls*g*pow(mp,3)*sin(psi)*cos(psi - 3*theta) - 2.0*La*Ls*g*pow(mp,3)*sin(psi)*cos(psi)*pow(cos(theta),3) + 2.5*La*Ls*g*pow(mp,3)*sin(psi)*cos(psi)*cos(theta) - 1.0*La*Ls*g*pow(mp,3)*pow(sin(theta),3) + 0.75*La*Ls*g*pow(mp,3)*sin(theta) - 0.25*La*Ls*g*pow(mp,3)*sin(3*theta) + 1.0*La*T_drive*pow(mp,2)*cos(psi) + 0.25*La*g*mb*pow(mp,2)*rc*sin(psi - 3*theta) + 1.0*La*g*mb*pow(mp,2)*rc*sin(psi - theta)*pow(sin(theta),2) - 0.25*La*g*mb*pow(mp,2)*rc*sin(psi + theta) - 0.111111111111111*pow(Ls,3)*T_drive*pow(ds,2) + 0.111111111111111*pow(Ls,3)*pow(ds,2)*g*mb*rc*sin(theta) - 0.666666666666667*pow(Ls,2)*T_drive*ds*mp + 0.666666666666667*pow(Ls,2)*ds*g*mb*mp*rc*sin(theta) - 1.0*Ls*T_drive*pow(mp,2) + 1.0*Ls*g*mb*pow(mp,2)*rc*sin(theta))/(Ls*(0.111111111111111*Ia*pow(Ls,2)*pow(ds,2) + 0.666666666666667*Ia*Ls*ds*mp + 1.0*Ia*pow(mp,2) + 0.0833333333333333*pow(La,2)*pow(Ls,3)*pow(ds,3)*pow(sin(psi),2) + 0.0277777777777778*pow(La,2)*pow(Ls,3)*pow(ds,3) + 0.583333333333333*pow(La,2)*pow(Ls,2)*pow(ds,2)*mp*pow(sin(psi),2) + 0.194444444444444*pow(La,2)*pow(Ls,2)*pow(ds,2)*mp + 1.33333333333333*pow(La,2)*Ls*ds*pow(mp,2)*pow(sin(psi),2) + 0.333333333333333*pow(La,2)*Ls*ds*pow(mp,2) + 1.0*pow(La,2)*pow(mp,3)*pow(sin(psi),2))));
}

double * dyn_ode(double *y, struct dyn_params dp, double T_drive){
 static double ydot[4]; //theta_dot, theta_dd, psi_dot, psi_dd
 
 ydot[0] = y[1];
 ydot[1] = thetadd(y, dp.La, dp.Ls, dp.ds, dp.mb, dp.rc, dp.Ia, dp.mp, dp.g, T_drive);
 ydot[2] = y[3];
 ydot[3] = psidd(y, dp.La, dp.Ls, dp.ds, dp.mb, dp.rc, dp.Ia, dp.mp, dp.g, T_drive);
 return ydot;
}

double * dyn_ode_trough(double *y, struct dyn_params dp, double T_drive){
 static double ydt[4];
 double It;
 double psi;
 psi = y[2];
 It =  dp.Ia + dp.Ls*dp.ds*(3*pow(dp.La,2) - 3*dp.La*dp.Ls*cos(psi) + pow(dp.Ls,2))/3 + dp.mp*(pow(dp.La,2) - 2*dp.La*dp.Ls*cos(psi) + pow(dp.Ls,2));
 ydt[0] = y[1];
 ydt[1] = -T_drive/It;
 ydt[2] = 0;
 ydt[3] = 0;
 return ydt;
}

double T(double *y, struct T_params tp){
 double m, theta, w, tau;
 theta = y[0];
 w = y[1];
 m = 1;
 tau = tp.c0 + tp.c1*theta + tp.c2*pow(theta, 2) + tp.c3*pow(theta,3) + tp.c4*pow(theta,4) + tp.c5*pow(theta,5) + tp.c6*pow(theta,6) + tp.c7*pow(theta, 7) + tp.c8*pow(theta,8) + tp.c9*pow(theta,9) + tp.c10*pow(theta,10);
 if ((w > 0)!=(tau < 0)){
  m = tp.kw; // allows for direction-dependent torque, but only linear scaling of entire torque function
  // this is appropriate for rubber pullers
  // w>0 corresponds to backdriving; backdriving XOR negative torque result in rubber stretching rather than contracting
  // note that positive torque causes negative angular acceleration because of the fucked up torque sign convention I use
 }
 return m*tau;
}

double * dyn_ode_wrapper(double *y, struct dyn_params dp, struct T_params tp){
 double T_drive;
 T_drive = T(y, tp);
 return dyn_ode(y, dp, T_drive);
}

double * dyn_ode_trough_wrapper(double *y, struct dyn_params dp, struct T_params tp){
 double T_drive;
 T_drive = T(y, tp);
 return dyn_ode_trough(y, dp, T_drive);
}

double *scalar_mult4(double *vector, double scalar){
 static double sm4_res[4];
 int i;
 for (i=0; i<=3; i++){
  sm4_res[i] = scalar*vector[i];
 }
 return sm4_res; // modifies 4-element vector in-place
}

double *vec_add4(double *vec1, double *vec2){
 static double va4_res[4];
 int i;
 for (i=0; i<=3; i++){
  va4_res[i] = vec1[i] + vec2[i];
 }
 return va4_res;
}

double * dyn_step_RK4(double dt, double *y, struct dyn_params dp, struct T_params tp){
 double k1[4];
 double k2[4];
 double k3[4];
 double k4[4];
 double *kt;
 double *ytemp;
 static double yfinal[4]; // DOESN'T PERSIST ON REPEAT CALL
 int i, j, k;

 kt = scalar_mult4(dyn_ode_wrapper(y, dp, tp), dt);
 for (i=0; i<=3; i++){
  k1[i] = kt[i];
 }
 kt = scalar_mult4(dyn_ode_wrapper((vec_add4(y, scalar_mult4(k1, 0.5))), dp, tp), dt);
 for (i=0; i<=3; i++){
  k2[i] = kt[i];
 }
 kt = scalar_mult4(dyn_ode_wrapper((vec_add4(y, scalar_mult4(k2, 0.5))), dp, tp), dt);
 for (i=0; i<=3; i++){
  k3[i] = kt[i];
 }
 kt = scalar_mult4(dyn_ode_wrapper((vec_add4(y, k3)), dp, tp), dt);
 for (i=0; i<=3; i++){
  k4[i] = kt[i];
 }

 /*FILE * fp;
 fp = fopen("dllog.txt","w");
 fprintf(fp, "k1 = ");
 for (j=0; j<=3; j++){
  fprintf(fp, "%f, ", k1[j]);
 }
 fprintf(fp, "k2 = ");
 for (j=0; j<=3; j++){
  fprintf(fp, "%f, ", k2[j]);
 }
 fprintf(fp, "k3 = ");
 for (j=0; j<=3; j++){
  fprintf(fp, "%f, ", k3[j]);
 }
 fprintf(fp, "k4 = ");
 for (j=0; j<=3; j++){
  fprintf(fp, "%f, ", k4[j]);
 }
 fclose(fp);*/

 ytemp = vec_add4(k1, scalar_mult4(k2, 2.0));
 ytemp = vec_add4(ytemp, scalar_mult4(k3, 2.0));
 ytemp = vec_add4(ytemp, scalar_mult4(k4, 1.0));
 ytemp = vec_add4(y, scalar_mult4(ytemp, (1.0/6.0)));
 for (i=0; i<=3; i++){
  yfinal[i] = ytemp[i];
 }
 // It is vastly important to remember that all these functions that return arrays are returning
 // pointers to a sort of global variable. If you want the results you get out to persist after
 // the function is called again, the results need to be copied.
 return yfinal;
}

double * dyn_step_trough_RK4(double dt, double *y, struct dyn_params dp, struct T_params tp){
 double k1[4];
 double k2[4];
 double k3[4];
 double k4[4];
 double *kt;
 double *ytemp;
 static double yft[4]; // DOESN'T PERSIST ON REPEAT CALL
 int i, j, k;

 kt = scalar_mult4(dyn_ode_trough_wrapper(y, dp, tp), dt);
 for (i=0; i<=3; i++){
  k1[i] = kt[i];
 }
 kt = scalar_mult4(dyn_ode_trough_wrapper((vec_add4(y, scalar_mult4(k1, 0.5))), dp, tp), dt);
 for (i=0; i<=3; i++){
  k2[i] = kt[i];
 }
 kt = scalar_mult4(dyn_ode_trough_wrapper((vec_add4(y, scalar_mult4(k2, 0.5))), dp, tp), dt);
 for (i=0; i<=3; i++){
  k3[i] = kt[i];
 }
 kt = scalar_mult4(dyn_ode_trough_wrapper((vec_add4(y, k3)), dp, tp), dt);
 for (i=0; i<=3; i++){
  k4[i] = kt[i];
 }

 ytemp = vec_add4(k1, scalar_mult4(k2, 2.0));
 ytemp = vec_add4(ytemp, scalar_mult4(k3, 2.0));
 ytemp = vec_add4(ytemp, scalar_mult4(k4, 1.0));
 ytemp = vec_add4(y, scalar_mult4(ytemp, (1.0/6.0)));
 for (i=0; i<=3; i++){
  yft[i] = ytemp[i];
 }

 return yft;
}
