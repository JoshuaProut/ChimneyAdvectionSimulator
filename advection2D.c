/*******************************************************************************
2D advection example program which advects a Gaussian u(x,y) at a fixed velocity



Outputs: initial.dat - inital values of u(x,y)
         final.dat   - final values of u(x,y)

         The output files have three columns: x, y, u

         Compile with: gcc -o advection2D -std=c99 advection2D.c -lm

Notes: The time step is calculated using the CFL condition

********************************************************************************/

/*********************************************************************
                     header files
**********************************************************************/

#include <stdio.h>
#include <math.h>
#include <omp.h>



/* Horizontal velocity function */

const float k = 0.41;
const float z0 = 1.0;
const float u = 0.2;

float horizontal_velocity (float z) {
    float hv;

    if (z > z0) {
        hv = ((u/k)*(float)log(z/z0))/((u/k) * fabs((float)log(30.0)));
    } else {
        hv = 0;
    }

    return hv;
}


/*********************************************************************
                      Main function
**********************************************************************/

int main() {

    /* Grid properties */
    const int NX = 1000;    // Number of x points
    const int NY = 1000;    // Number of y points
    const float xmin = 0.0; // Minimum x value
    const float xmax = 30.0; // Maximum x value
    const float ymin = 0.0; // Minimum y value
    const float ymax = 30.0; // Maximum y value

    /* Parameters for the Gaussian initial conditions */
    const float x0 = 3.0;                    // Centre(x)
    const float y0 = 15.0;                    // Centre(y)
    const float sigmax = 1.0;               // Width(x)
    const float sigmay = 5.0;               // Width(y)
    const float sigmax2 = sigmax * sigmax; // Width(x) squared
    const float sigmay2 = sigmay * sigmay; // Width(y) squared

    /* Boundary conditions */
    const float bval_left = 0.0;    // Left boudnary value
    const float bval_right = 0.0;   // Right boundary value
    const float bval_lower = 0.0;   // Lower boundary
    const float bval_upper = 0.0;   // Upper bounary

    /* Time stepping parameters */
    const float CFL = 0.9;   // CFL number
    const int nsteps = 800; // Number of time steps

    /* Velocity */
    const float velx = 1.0; // Velocity in x direction
    const float vely = 0.00; // Velocity in y direction

    /* Arrays to store variables. These have NX+2 elements
       to allow boundary values to be stored at both ends */
    float x[NX + 2];          // x-axis values
    float y[NX + 2];          // y-axis values
    float u[NX + 2][NY + 2];    // Array of u values
    float dudt[NX + 2][NY + 2]; // Rate of change of u

    float x2;   // x squared (used to calculate iniital conditions)
    float y2;   // y squared (used to calculate iniital conditions)

    /* Calculate distance between points */
    float dx = (xmax - xmin) / ((float) NX);
    float dy = (ymax - ymin) / ((float) NY);

    /* Calculate time step using the CFL condition */
    /* The fabs function gives the absolute value in case the velocity is -ve */
    float dt = CFL / ((fabs(velx) / dx) + (fabs(vely) / dy));

    /*** Report information about the calculation ***/
    printf("Grid spacing dx     = %g\n", dx);
    printf("Grid spacing dy     = %g\n", dy);
    printf("CFL number          = %g\n", CFL);
    printf("Time step           = %g\n", dt);
    printf("No. of time steps   = %d\n", nsteps);
    printf("End time            = %g\n", dt * (float) nsteps);
    printf("Distance advected x = %g\n", velx * dt * (float) nsteps);
    printf("Distance advected y = %g\n", vely * dt * (float) nsteps);

    /*** Place x points in the middle of the cell ***/
#pragma omp parallel for default (none) shared (x, dx)
    for (int i = 0; i < NX + 2; i++) {
        x[i] = ((float) i - 0.5) * dx;
    }

    /*** Place y points in the middle of the cell ***/
#pragma omp parallel for default (none) shared (y, dy)
    for (int j = 0; j < NY + 2; j++) {
        y[j] = ((float) j - 0.5) * dy;
    }

    /*** Set up Gaussian initial conditions ***/
#pragma omp parallel for private (x2, y2) shared (u,x,y) collapse (2)
    for (int i = 0; i < NX + 2; i++) {
        for (int j = 0; j < NY + 2; j++) {
            x2 = (x[i] - x0) * (x[i] - x0);
            y2 = (y[j] - y0) * (y[j] - y0);
            u[i][j] = exp(-1.0 * ((x2 / (2.0 * sigmax2)) + (y2 / (2.0 * sigmay2))));
        }
    }

    /*** Write array of initial u values out to file ***/
    FILE *initialfile;
    initialfile = fopen("initial.dat", "w");
    for (int i = 0; i < NX + 2; i++) {
        for (int j = 0; j < NY + 2; j++) {
            fprintf(initialfile, "%g %g %g\n", x[i], y[j], u[i][j]);
        }
    }
    fclose(initialfile);

    /*** Update solution by looping over time steps ***/
    for (int m = 0; m < nsteps; m++) {

        /*** Apply boundary conditions at u[0][:] and u[NX+1][:] ***/
#pragma omp parallel for default (none) shared (u)
        for (int j = 0; j < NY + 2; j++) {
            u[0][j] = bval_left;
            u[NX + 1][j] = bval_right;
        }

        /*** Apply boundary conditions at u[:][0] and u[:][NY+1] ***/
#pragma omp parallel for default (none) shared (u)
        for (int i = 0; i < NX + 2; i++) {
            u[i][0] = bval_lower;
            u[i][NY + 1] = bval_upper;
        }

        /*** Calculate rate of change of u using leftward difference ***/
#pragma omp parallel for shared (dudt) collapse (2)
        for (int i = 1; i < NX + 1; i++) {
            for (int j = 1; j < NY + 1; j++) {
                dudt[i][j] = -horizontal_velocity(y[j]) * (u[i][j] - u[i - 1][j]) / dx
                             - vely * (u[i][j] - u[i][j - 1]) / dy;
            }
        }

        /*** Update u from t to t+dt ***/
#pragma omp parallel for shared (u) collapse (2)
        for (int i = 1; i < NX + 1; i++) {
            for (int j = 1; j < NY + 1; j++) {
                u[i][j] = u[i][j] + dudt[i][j] * dt;
            }
        }

    } // time loop

    /*** Write array of final u values out to file ***/
    FILE *finalfile;
    finalfile = fopen("final.dat", "w");
    for (int i = 0; i < NX + 2; i++) {
        for (int j = 0; j < NY + 2; j++) {
            fprintf(finalfile, "%g %g %g\n", x[i], y[j], u[i][j]);
        }
    }
    fclose(finalfile);


    /* Calculates averages for task 2.4 */
    float values_to_plot[NX];
    for (int i=1; i<NX; i++){
        float count = 0.0;
        float average = 0.0;

        for (int j=1; j<NY; j++){
            count += u[i][j];
        }
        average = count / NX;
        values_to_plot[i] = average;
    }

    /* Writes averages to file */
    FILE *finalfile2;
    finalfile2 = fopen("plot.dat", "w");
    for (int i = 0; i < NX + 2; i++) {
        for (int j = 0; j < NY + 2; j++) {
            fprintf(finalfile2, "%g %g \n", x[i], values_to_plot[i]);
        }
    }
    fclose(finalfile2);


    return 0;
}

