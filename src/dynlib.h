#ifndef _DYNLIB_H_
#define _DYNLIB_H_

struct dyn_params
{
 double La, Ls, ds, mb, rc, Ia, mp, g;
};

struct T_params
{
 double kw, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10;
};

extern double * dyn_ode(double *y, struct dyn_params dp, double T_drive);

extern double * dyn_ode_trough(double *y, struct dyn_params dp, double T_drive);

extern double T(double *y, struct T_params tp);

extern double * dyn_step_RK4(double dt, double *y, struct dyn_params dp, struct T_params tp); //single step RK4 for dynamics given by dyn_ode
extern double * dyn_step_trough_RK4(double dt, double *y, struct dyn_params dp, struct T_params tp);
//other functions are strictly internal, not called from outside

#endif
