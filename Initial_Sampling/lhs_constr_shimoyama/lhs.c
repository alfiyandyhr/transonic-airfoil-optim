/* Latin Hypercube Sampling (LHS) */
/* coded by Koji Shimoyama */
/*          12/01/2021 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

long idum;

/* generate a random number */
#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define NTAB 32
#define NDIV (1+(IM-1)/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)
double ran1(long *idum)
{
  int j;
  long k;
  static long iy = 0;
  static long iv[NTAB];
  double temp;

  if (*idum <= 0 || !iy) {
    if (-(*idum) < 1) *idum = 1;
    else *idum = -(*idum);
    for (j=NTAB+7; j>=0; j--) {
      k = (*idum)/IQ;
      *idum = IA*(*idum-k*IQ)-IR*k;
      if (*idum < 0) *idum += IM;
      if (j < NTAB) iv[j] = *idum;
    }
    iy = iv[0];
  }
  k = (*idum)/IQ;
  *idum = IA*(*idum-k*IQ)-IR*k;
  if (*idum < 0) *idum += IM;
  j = iy/NDIV;
  iy = iv[j];
  iv[j] = *idum;
  if ((temp = AM*iy) > RNMX) return RNMX;
  else return temp;
}

/* !!! START for pre-evaluation !!! */
/* evaluate functions */
int fval(int nx, int nf, int ng, double *x, double *f, double *g)
{
  FILE *fp;
  int n;
  double dmy;

  /* fp = fopen("eval.in", "w"); */
  fp = fopen("preeval.in", "w");
  fprintf(fp, "%d %d %d %d\n", 1, nx, nf, ng);
  for (n=0; n<nx; n++)
    fprintf(fp, "%lf ", x[n]);
  fclose(fp);
  /* system("./eval"); */
  system("python preeval.py");
  /* fp = fopen("eval.out", "r"); */
  fp = fopen("preeval.out", "r");
  for (n=0; n<nx; n++)
    fscanf(fp, "%lf", &dmy);
  for (n=0; n<nf; n++)
    fscanf(fp, "%lf", &f[n]);
  for (n=0; n<ng; n++)
    fscanf(fp, "%lf", &g[n]);
  fclose(fp);

  return 0;
}
/* !!! END for pre-evaluation !!! */

int main(void)
{
  FILE *fp;
  int n, m, l;
  int smp, smpmax, ndv, ip, **ipd, ipdtmp, *iaux;
  double *dvmin, *dvmax, **dv;
  /* !!! START for pre-evaluation !!! */
  int pev, pevmax, iffeas, nobj, ncon;
  double *dvevl, *objevl, *conevl;
  /* !!! END for pre-evaluation !!! */

  remove("preeval.in");
  remove("preeval.out");


  // /* input parameters */
  // printf("input sample size:\n");
  // scanf("%d", &smpmax);
  // if (smpmax < 1) {
  //   printf("!!!!!! ERROR in main: smpmax must be >= 1 !!!!!!\n");
  //   exit(1);
  // }
  // /* !!! START for pre-evaluation !!! */
  // printf("input the maximum number of pre-evaluations per sample:\n");
  // scanf("%d", &pevmax);
  // if (pevmax < 1) {
  //   printf("!!!!!! ERROR in main: pevmax must be >= 1 !!!!!!\n");
  //   exit(1);
  // }
  // /* !!! END for pre-evaluation !!! */
  // printf("input the number of design variables:\n");
  // scanf("%d", &ndv);
  // if (ndv < 1) {
  //   printf("!!!!!! ERROR in main: ndv must be >= 1 !!!!!!\n");
  //   exit(1);
  // }

  fp = fopen("lhs_constr_config.in","r");
  fscanf(fp,"%d", &smpmax);
  fscanf(fp,"%d", &pevmax);
  fscanf(fp,"%d", &ndv);
  fscanf(fp,"%d", &nobj);
  fscanf(fp,"%d", &ncon);

  dvmin = (double *)malloc(ndv*sizeof(double));
  dvmax = (double *)malloc(ndv*sizeof(double));
  dv = (double **)malloc(ndv*sizeof(double *));
  for (n=0; n<ndv; n++)
    dv[n] = (double *)malloc(smpmax*sizeof(double));
  ipd = (int **)malloc(ndv*sizeof(double *));
  for (n=0; n<ndv; n++)
    ipd[n] = (int *)malloc(smpmax*sizeof(double));
  iaux = (int *)malloc(ndv*sizeof(double));
  
  /* !!! START for pre-evaluation !!! */
  dvevl = (double *)malloc(ndv*sizeof(double));
  objevl = (double *)malloc(nobj*sizeof(double));
  conevl = (double *)malloc(ncon*sizeof(double));
  /* !!! END for pre-evaluation !!! */

  for (n=0; n<ndv; n++) {
    // printf("input the minimum value of design variable %d:\n", n+1);
    fscanf(fp, "%lf %lf", &dvmin[n], &dvmax[n]);
    // printf("input the maximum value of design variable %d:\n", n+1);
    if (dvmin[n] >= dvmax[n]) {
      printf("!!!!!! ERROR in main: dvmin must be < dvmax !!!!!!\n");
      exit(1);
    }
  }

  fclose(fp);
  
  printf("input a random number seed (negative integer to initialize):\n");
  scanf("%ld", &idum);

  /* initialize */
  smp = 0;
  ip = 0;
  for (n=0; n<ndv; n++)
    for (m=0; m<smpmax; m++)
      ipd[n][m] = m;

  /* generate samples */
  while (smp < smpmax) {
    /* !!! START for pre-evaluation !!! */
    iffeas = 0;
    pev = 0;
    while ((! iffeas) && (pev < pevmax)) {
    /* !!! END for pre-evaluation !!! */
      if (ip > smpmax-1)
	ip = 0;
      for (n=0; n<ndv; n++) {
	iaux[n] = ip+(int)(ran1(&idum)*(double)(smpmax-ip));
	iaux[n] = MIN(iaux[n], smpmax-1);
	dv[n][smp] = ((double)ipd[n][iaux[n]]+ran1(&idum))/(double)smpmax;
      }
      /* !!! START for pre-evaluation !!! */
      printf("pre-evaluation = %d\n", pev+1);
      for (n=0; n<ndv; n++)
	dvevl[n] = dvmin[n]+dv[n][smp]*(dvmax[n]-dvmin[n]);
      fval(ndv, nobj, ncon, dvevl, objevl, conevl);
      iffeas = 1;
      for (n=0; n<ncon; n++)
	if (conevl[n] > 0.0) {
	  iffeas = 0;
	  pev++;
	  break;
	}
    }
    if (! iffeas) {
      printf("!!!!!! ERROR in main: feasible sample = %d is not found !!!!!!\n", smp+1);
      break;
    }
    /* !!! END for pre-evaluation !!! */
    for (n=0; n<ndv; n++) {
      ipdtmp = ipd[n][iaux[n]];
      ipd[n][iaux[n]] = ipd[n][ip];
      ipd[n][ip] = ipdtmp;
    }
    ip++;

    printf("sample = %d\n", smp+1);
    smp++;
  }

  /* check orthogonality */
  for (n=0; n<ndv; n++)
    for (m=0; m<smp-1; m++) 
      for (l=m+1; l<smp; l++)
	if (ipd[n][m] == ipd[n][l]) {
	  printf("!!!!!! ERROR in main: samples %d and %d are not orthogonal !!!!!!\n", m+1, l+1);
	  exit(1);
	}

  /* output data */
  fp = fopen("dv_lhs.dat", "w");
  for (n=0; n<smp; n++) {
    for (m=0; m<ndv; m++)
      fprintf(fp, "%lf ", dvmin[m]+dv[m][n]*(dvmax[m]-dvmin[m]));
    fprintf(fp, "\n");
  }
  fclose(fp);

  free(dvmin);
  free(dvmax);
  free(dv);
  free(ipd);
  free(iaux);
  /* !!! START for pre-evaluation !!! */
  free(dvevl);
  free(objevl);
  free(conevl);
  /* !!! END for pre-evaluation !!! */

  return 0;
}
