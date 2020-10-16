#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define STR_MAX_LEN 256
#define frand() (rand()/(RAND_MAX+1.0)) 

void setting(int *l, double *rx, double *ry, double *rz, int n)
{
  int i, rem, nbs, r, *bas; 

  *l = (int)sqrt((double)n); /*what is l? square root for Natom*/
  if((*l)*(*l) < n) *l += 1;


  srand(time(NULL));

  nbs = (*l)*(*l);
  bas = (int *)malloc(nbs*sizeof(int));
  for(i = 0; i < nbs; i++) bas[i] = i;
  rem = nbs - n;
  while(rem > 0){
    r = (int)(nbs*frand());
    for(i = r; i < nbs-1; i++) bas[i] = bas[i+1];
    nbs--;
    rem--;
  }

  for(i = 0; i < n; i++){
    rx[i] = bas[i]%(*l);
    ry[i] = bas[i]/(*l);
    rz[i] = (*l)/2;
    rx[i] += 0.05*sqrt(-2.0*log(frand()))*cos(2.0*M_PI*frand()); //Box Muller Transform
    ry[i] += 0.05*sqrt(-2.0*log(frand()))*cos(2.0*M_PI*frand());
    rz[i] += 0.05*sqrt(-2.0*log(frand()))*cos(2.0*M_PI*frand());
    if(rx[i] < 0.0) rx[i] += *l;
    if(ry[i] < 0.0) ry[i] += *l;
    if(rz[i] < 0.0) rz[i] += *l;
    if(rx[i] >  *l) rx[i] -= *l;
    if(ry[i] >  *l) ry[i] -= *l;
    if(rz[i] >  *l) rz[i] -= *l;
  }

  free(bas);
}

void force(double *fx, double *fy, double *fz,
  int n, int l, double *rx, double *ry, double *rz)
{
  int i, j;
  double dx, dy, dz, dr;
  double rc, fr;

  rc = l/2.0;
#ifdef _OPENMP
  #pragma omp for
#endif
  for(i = 0; i < n; i++){
    fx[i] = 0.0;
    fy[i] = 0.0;
    fz[i] = 0.0;
  }

#ifdef _OPENMP
  #pragma omp for
#endif
  for(i = 0; i < n; i++){
    for(j = 0; j < n; j++){
      if(i == j) continue; 
      dx = rx[i] - rx[j];
      dy = ry[i] - ry[j];
      dz = rz[i] - rz[j];
      if(dx < -rc) dx += l;
      if(dy < -rc) dy += l;
      if(dz < -rc) dz += l;
      if(dx >  rc) dx -= l;
      if(dy >  rc) dy -= l;
      if(dz >  rc) dz -= l;
      dr = sqrt(dx*dx + dy*dy + dz*dz); //Resultan
      if(dr < rc){
        fr = 1.0/pow(dr,13) - 1.0/pow(dr,7); //Force: derivation of Potential En. (Lennard-Jones)
        fx[i] += fr*dx/dr;
        fy[i] += fr*dy/dr;
        fz[i] += fr*dz/dr;
      }
    }
  }
}

void outdata(FILE *fp, int n, int l,
  double *rx, double *ry, double *rz, double *vx, double *vy, double *vz, int step)
{
  int i, j;
  double rc;
  double dx, dy, dz, dr;  
  double et, ek, ev;  

  rc = l/2.0;
  ek = 0.0;
  ev = 0.0;
  for(i = 0; i < n; i++){
    ek += (vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i])/2.0; 
    for(j = i+1; j < n; j++){
      dx = rx[i] - rx[j];
      dy = ry[i] - ry[j];
      dz = rz[i] - rz[j];

      if(dx < -rc) dx += l;
      if(dy < -rc) dy += l;
      if(dz < -rc) dz += l;
      if(dx >  rc) dx -= l;
      if(dy >  rc) dy -= l;
      if(dz >  rc) dz -= l;
      dr = sqrt(dx*dx + dy*dy + dz*dz);
      if(dr < rc) ev += (1.0/pow(dr,12) - 2.0/pow(dr,6))/12.0; //Lennard-Jones
    }
  }
  et = ek + ev;

  fprintf(fp, "%i %g %g %g\n", step, ek, ev, et);
}

void outaxsf(FILE *gp, int n, int l, double dt,
  double *rx, double *ry, double *rz, double *fx, double *fy, double *fz, int step)
{
  int i;

  fprintf(gp, "PRIMCOORD %i\n", step+1);
  fprintf(gp, "%i 1\n", n);

  for(i = 0; i < n; i++)
    fprintf(gp, "%g %g %g \n",rx[i], ry[i], rz[i]);

  fflush(gp);
}

int main(void)
{
//  clock_t begin = clock();

  char str[STR_MAX_LEN];
  int i, step;
  int l, na, nstep;
  double dt;
  double *rx, *ry, *rz, *vx, *vy, *vz, *fx, *fy, *fz, *ax, *ay, *az;
  double m = 100;
  FILE *fp, *gp ,*ip, *op, *input;

  
  input=fopen("md.in", "r");
  for(i = 0; fgets(str, STR_MAX_LEN, input) != NULL; i++){
    if(*str == '#') fgets(str, STR_MAX_LEN, input);
    switch(i){
      case 0:
        sscanf(str, "%i", &na);
        break;
      case 1:
        sscanf(str, "%i", &nstep);
        break;
      case 2:
        sscanf(str, "%lg", &dt);
        break;
      default:
        printf("error in md.in\n");
        return 0;
    }
  }
  fclose(input);
  fp=fopen("md_velet_energy.dat", "w");
  gp=fopen("md_velet_data.dat", "w");
  ip=fopen("init_pos_velet.dat", "w");
  op=fopen("final_pos_velet.dat", "w");

  rx=(double *)malloc(na*sizeof(double));
  ry=(double *)malloc(na*sizeof(double));
  rz=(double *)malloc(na*sizeof(double));
  vx=(double *)malloc(na*sizeof(double));
  vy=(double *)malloc(na*sizeof(double));
  vz=(double *)malloc(na*sizeof(double));
  fx=(double *)malloc(na*sizeof(double));
  fy=(double *)malloc(na*sizeof(double));
  fz=(double *)malloc(na*sizeof(double));
  ax = (double *)malloc(na*sizeof(double));
  ay = (double *)malloc(na*sizeof(double));
  az = (double *)malloc(na*sizeof(double));

  setting(&l, rx, ry, rz, na);

  for(i = 0; i < na; i++){
  fprintf(ip, "%f %f %f\n", rx[i], ry[i], rz[i]);
  }

  for(i = 0; i < na; i++){
    vx[i] = 0.0;
    vy[i] = 0.0;
    vz[i] = 0.0;
  }
#ifdef _OPENMP
  #pragma omp parallel
#endif
  { force(fx, fy, fz, na, l, rx, ry, rz); }
  outdata(fp, na, l, rx, ry, rz, vx, vy, vz, 0);

  for(step = 1; step < nstep; step++){
    printf("step = %d\r",step);    
    fflush(stdout);
    for(i = 0; i < na; i++){
        ax[i] = fx[i] / m;
        ay[i] = fy[i] / m;
        az[i] = fz[i] / m;

        rx[i] += vx[i] * dt + 0.5 * ax[i] * dt * dt;
        ry[i] += vy[i] * dt + 0.5 * ay[i] * dt * dt;
        rz[i] += vz[i] * dt + 0.5 * az[i] * dt * dt;

        if(rx[i] < 0.0) rx[i] += l;
        if(ry[i] < 0.0) ry[i] += l;
        if(rz[i] < 0.0) rz[i] += l;
        if(rx[i] >   l) rx[i] -= l;
        if(ry[i] >   l) ry[i] -= l;
        if(rz[i] >   l) rz[i] -= l;

    }

    for(i = 0; i < na; i++){
     
        vx[i] += (0.5 * ax[i] * dt);
        vy[i] += (0.5 * ay[i] * dt);
        vz[i] += (0.5 * az[i] * dt);
    }

#ifdef _OPENMP
    #pragma omp parallel
#endif
    { force(fx, fy, fz, na, l, rx, ry, rz); }

    for(i = 0; i < na; i++){

        ax[i] = fx[i] / m;
        ay[i] = fy[i] / m;
        az[i] = fz[i] / m;

        vx[i] +=  (0.5 * ax[i] * dt);
        vy[i] +=  (0.5 * ay[i] * dt);
        vz[i] +=  (0.5 * az[i] * dt);
    }
  
       outdata(fp, na, l, rx, ry, rz, vx, vy, vz, step);
       outaxsf(gp, na, l, dt, rx, ry, rz, fx, fy, fz, step); 
  }
  for(i = 0; i < na; i++){
  fprintf(op, "%f %f %f\n", rx[i], ry[i], rz[i]);
 
  }
  
  free(rx);
  free(ry);
  free(rz);
  free(vx);
  free(vy);
  free(vz);
  free(fx);
  free(fy);
  free(fz);

  fclose(fp);



  return 0;
}
