#include <stdio.h>

int main(void)
{
	FILE *fp;
	double x,y;
	fp = fopen("test.in","r");
	fscanf(fp,"%lf %lf",&x,&y);
	

	printf("%lf %lf\n", x,y);

	fclose(fp);
return 0;

}