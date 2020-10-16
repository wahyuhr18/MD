#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <helper_cuda.h>
#include <helper_cuda_drvapi.h>
#include <helper_functions.h>
#include <helper_image.h>
#include <helper_math.h>
#include <helper_string.h>
#include <helper_timer.h>

#define PI 3.14159265358979323846
#define STR_MAX_LEN 256
#define frand() (rand()/(RAND_MAX+1.0))

void setting(int *l, float *rx, float *ry, float *rz, int n)
{
  int i, rem, nbs, r, *bas; 

  *l = (int)sqrt((float)n);
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

void outdata(FILE *fp, int n, int l,
  float *rx, float *ry, float *rz, float *vx, float *vy, float *vz, int step)
{
  int i, j;
  float rc;
  float dx, dy, dz, dr;  
  float et, ek, ev;  

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

  fprintf(fp, "%i %f %f %f\n", step, ek, ev, et);
}

void outaxsf(FILE *gp, int n, int l, float dt,
  float *rx, float *ry, float *rz, float *fx, float *fy, float *fz, int nstep, int step)
{
  int i;

  if(step == 0){
    fprintf(gp, "ANIMSTEPS %i\n", nstep);
    fprintf(gp, "SLAB\n");
    fprintf(gp, "PRIMVEC\n");
    fprintf(gp, "%f 0.0 0.0\n", (float)l);
    fprintf(gp, "0.0 %f 0.0\n", (float)l);
    fprintf(gp, "0.0 0.0 1.0\n");
  }
  fprintf(gp, "PRIMCOORD %i\n", step+1);
  fprintf(gp, "%i 1\n", n);

  for(i = 0; i < n; i++){
    fprintf(gp, "%g %g %g\n", rx[i], ry[i], rz[i]);
  }

  fflush(gp);
}

__global__ void force(int thr, int n, int l,
  float *fx, float *fy, float *fz, float *rx, float *ry, float *rz)
{
  int i, j;
  float dx, dy, dz, dr;
  float rc, fr;
#ifdef _SHR
  int k, blk;
  __shared__ float shrx[64], shry[64], shrz[64];
#endif

  if((i = thr*blockIdx.x * blockDim.x + threadIdx.x) > n) return;

  rc = l/2.0;
  fx[i] = 0.0;
  fy[i] = 0.0;
  fz[i] = 0.0;
#ifdef _SHR
  blk = n/thr + (n%thr ? 1 : 0);

  for(k = 0; k < blk; k++){
    __syncthreads();   
    shrx[threadIdx.x] = rx[thr*k+threadIdx.x];
    shry[threadIdx.x] = ry[thr*k+threadIdx.x];
    shrz[threadIdx.x] = rz[thr*k+threadIdx.x];
    __syncthreads();   
    for(j = 0; j < thr; j++){
      if(i == thr*k+j) continue;
      dx = rx[i] - shrx[j];
      dy = ry[i] - shry[j];
      dz = rz[i] - shrz[j];
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
#else
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
    dr = sqrt(dx*dx + dy*dy + dz*dz);
    if(dr < rc){
      fr = 1.0/pow(dr,13) - 1.0/pow(dr,7);
      fx[i] += fr*dx/dr;
      fy[i] += fr*dy/dr;
      fz[i] += fr*dz/dr;
    }
  }
#endif
}

int main(int argc, char **argv)
{
  char str[STR_MAX_LEN];
  int i, step, size;
  int l, na, nstep, blk, thr;
  float dt;
  float *rx, *ry, *rz, *vx, *vy, *vz, *fx, *fy, *fz, *ax, *ay, *az;
  float *dev_rx, *dev_ry, *dev_rz, *dev_fx, *dev_fy, *dev_fz;
  float m = 100;
  FILE *fp,*ip, *op, *gp, *input;

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
        sscanf(str, "%g", &dt);
        break;
      default:
        printf("error in md.in\n");
        return 0;
    }
  }
  
  fclose(input);

  if(argc == 2){
    thr = atoi(argv[1]);
    if(thr > na) thr = na;
    blk = na/thr + (na%thr ? 1 : 0);
  } else{
    printf("error\n");
    return 1;
  }

 fp=fopen("md_cuda_leap_energy.dat", "w");
 gp=fopen("md_cuda_leap_data.dat", "w");
 ip=fopen("init_pos_cuda_leap.dat", "w");
 op=fopen("final_pos_leap_cuda.dat", "w");

  size = na*sizeof(float);

  rx=(float *)malloc(size);
  ry=(float *)malloc(size);
  rz=(float *)malloc(size);
  vx=(float *)malloc(size);
  vy=(float *)malloc(size);
  vz=(float *)malloc(size);
  fx=(float *)malloc(size);
  fy=(float *)malloc(size);
  fz=(float *)malloc(size);
  ax=(float *)malloc(size);
  ay=(float *)malloc(size);
  az=(float *)malloc(size);

  setting(&l, rx, ry, rz, na);
  
  for(i = 0; i < na; i++){
  fprintf(ip, "%f %f %f\n", rx[i], ry[i], rz[i]);
  } 

  for(i = 0; i < na; i++){
    vx[i] = 0.0;
    vy[i] = 0.0;
    vz[i] = 0.0;
  }

  cudaMalloc((void **)&dev_rx, size);
  cudaMalloc((void **)&dev_ry, size);
  cudaMalloc((void **)&dev_rz, size);
  cudaMalloc((void **)&dev_fx, size);
  cudaMalloc((void **)&dev_fy, size);
  cudaMalloc((void **)&dev_fz, size);

  cudaMemcpy(dev_rx, rx, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_ry, ry, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_rz, rz, size, cudaMemcpyHostToDevice);

  force<<<blk,thr>>>(thr, na, l, dev_fx, dev_fy, dev_fz, dev_rx, dev_ry, dev_rz);

  cudaDeviceSynchronize();

  cudaMemcpy(fx, dev_fx, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(fy, dev_fy, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(fz, dev_fz, size, cudaMemcpyDeviceToHost);

  outdata(fp, na, l, rx, ry, rz, vx, vy, vz, 0);

  for(step = 1; step < nstep; step++){
    printf("step = %d\r",step);    
    fflush(stdout);
    for(i = 0; i < na; i++){
        ax[i] = fx[i] / m;
        ay[i] = fy[i] / m;
        az[i] = fz[i] / m;

        vx[i] += 0.5 * ax[i] * dt;
        vy[i] += 0.5 * ay[i] * dt;
        vz[i] += 0.5 * az[i] * dt;
    }

    for(i = 0; i < na; i++){
     
        rx[i] += vx[i] * dt;
        ry[i] += vy[i] * dt;
        rz[i] += vz[i] * dt;

        if(rx[i] < 0.0) rx[i] += l;
        if(ry[i] < 0.0) ry[i] += l;
        if(rz[i] < 0.0) rz[i] += l;
        if(rx[i] >   l) rx[i] -= l;
        if(ry[i] >   l) ry[i] -= l;
        if(rz[i] >   l) rz[i] -= l;
    }
    cudaMemcpy(dev_rx, rx, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ry, ry, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_rz, rz, size, cudaMemcpyHostToDevice);

    force<<<blk,thr>>>(thr, na, l, dev_fx, dev_fy, dev_fz, dev_rx, dev_ry, dev_rz);

    cudaDeviceSynchronize();

    cudaMemcpy(fx, dev_fx, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(fy, dev_fy, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(fz, dev_fz, size, cudaMemcpyDeviceToHost);
    
    for(i = 0; i < na; i++){

      ax[i] = fx[i] / m;
      ay[i] = fy[i] / m;
      az[i] = fz[i] / m;

      vx[i] += (0.5 * ax[i] * dt);
      vy[i] += (0.5 * ay[i] * dt);
      vz[i] += (0.5 * az[i] * dt);
    }

    outdata(fp, na, l, rx, ry, rz, vx, vy, vz, step);
    outaxsf(gp, na, l, dt, rx, ry, rz, fx, fy, fz, nstep, nstep);
  }
  for(i = 0; i < na; i++){
  fprintf(op, "%f %f %f\n", rx[i], ry[i], rz[i]);
  }

  cudaFree(&dev_rx);
  cudaFree(&dev_ry);
  cudaFree(&dev_rz);
  cudaFree(&dev_fx);
  cudaFree(&dev_fy);
  cudaFree(&dev_fz);

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


