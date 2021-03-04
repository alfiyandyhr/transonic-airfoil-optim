#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void)
{
  FILE *fin, *fou;
  int n, m, ne, nx, nf, ng;
  double *x, *f, *g;
  fin = fopen("preeval.in", "r");
  fou = fopen("preeval.out", "w");
  fscanf(fin, "%d %d %d %d", &ne, &nx, &nf, &ng);

  x = (double *)malloc(nx*sizeof(double));
  f = (double *)malloc(nf*sizeof(double));
  g = (double *)malloc(ng*sizeof(double));

  for (m=0; m<ne; m++) {
    for (n=0; n<nx; n++)
      fscanf(fin, "%lf", &x[n]);

    f[0] = 12345;
    g[0] = x[0] - x[1];
    g[1] = -1;
    g[2] = -1;
    g[3] = -1;

    for (n=0; n<nx; n++)
      fprintf(fou, "%lf ", x[n]);
    for (n=0; n<nf; n++)
      fprintf(fou, "%lf ", f[n]);
    for (n=0; n<ng; n++)
      fprintf(fou, "%lf ", g[n]);
    fprintf(fou, "\n");
  }

  fclose(fin);
  fclose(fou);

  free(x);
  free(f);
  free(g);

  return 0;
}
