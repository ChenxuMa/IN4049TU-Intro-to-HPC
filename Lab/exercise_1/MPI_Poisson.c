#include "mpi.h"
//#include "mpi.h"
#include <stdio.h>
//#include <stdafx.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#define DEBUG 0

#define max(a,b) ((a)>(b)?a:b)

enum
{
    X_DIR, Y_DIR
};

/* global variables */
int gridsize[2];
double precision_goal;		/* precision_goal of solution */
int max_iter;			/* maximum number of iterations alowed */
int proc_rank;// could not sure of the type of the variable!!!
double wtime;
int np, proc_rank;
double wtime;
int proc_coord[2];
int offset[2];
unsigned long long total_data_communicated=0;
MPI_Datatype border_type[2];
int proc_top, proc_right, proc_bottom, proc_left;
int P;
int P_grid[2]; /*Processgrid dimension*/
int global_parity;
MPI_Comm grid_comm;
MPI_Status status;
double border_exchange_start_point, border_exchange_end_point;
double total_communication_time=0;

/* benchmark related variables */
clock_t ticks;			/* number of systemticks */
int timer_on = 0;		/* is timer running? */

/* local grid related variables */
double **phi;			/* grid */
int **source;			/* TRUE if subgrid element is a source */
int dim[2];			/* grid dimensions */

void Setup_Grid();
void Setup_Proc_Grid(int argc, char **pString);
void Setup_MPI_Datatypes();
void Exchange_Borders();
double Do_Step(int parity);
void Solve();
void Write_Grid();
void Clean_Up();
void Debug(char *mesg, int terminate);
void start_timer();
void resume_timer();
void stop_timer();
void print_timer();
void Merge_files(int P);

void start_timer()
{
    if (!timer_on)
    {
        MPI_Barrier(MPI_COMM_WORLD);

        ticks = clock();

        wtime=MPI_Wtime();
        timer_on = 1;
    }
}


void resume_timer()
{
    if (!timer_on)
    {
        ticks = clock() - ticks;
        wtime=MPI_Wtime()-wtime;
        timer_on = 1;
    }
}

void stop_timer()
{
    if (timer_on)
    {
        ticks = clock() - ticks;
        wtime=MPI_Wtime()-wtime;
        timer_on = 0;
    }
}

void print_timer()
{
    if (timer_on)
    {
        stop_timer();
        printf("(%i) Elasped time: %14.6f s (%5.1f%% CPU) \n"
                , proc_rank, wtime, 100.0*ticks*(1.0/CLOCKS_PER_SEC)/wtime);
        //printf("Elapsed processortime: %14.6f s\n", ticks * (1.0 / CLOCKS_PER_SEC));
        resume_timer();
    }
    else
        printf("(%i) Elasped time: %14.6f s (%5.1f%% CPU) \n"
                , proc_rank, wtime, 100.0*ticks*(1.0/CLOCKS_PER_SEC)/wtime);
    //printf("Elapsed processortime: %14.6f s\n", ticks * (1.0 / CLOCKS_PER_SEC));
}

void Debug(char *mesg, int terminate)
{
    if (DEBUG || terminate)
        printf("%s\n", mesg);
    if (terminate)
        exit(1);
}
void Setup_Proc_Grid(int argc, char **argv) {
    int wrap_around[2];
    int reorder;
    Debug("My_MPI_Init", 0);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    if(argc>2){
        P_grid[X_DIR]=atoi(argv[1]);
        P_grid[Y_DIR]=atoi(argv[2]);
        if(P_grid[X_DIR]*P_grid[Y_DIR]!=P){
            Debug("ERROR: Process grid dimension do not match with P", 1);
        }
    }else{
        Debug("ERROR: Wrong parameter input", 1);
    }
    wrap_around[X_DIR]=0;
    wrap_around[Y_DIR]=0;
    reorder=1;
    MPI_Cart_create(MPI_COMM_WORLD, 2, P_grid, wrap_around, reorder, &grid_comm);
    MPI_Comm_rank(grid_comm, &proc_rank);
    MPI_Cart_coords(grid_comm, proc_rank, sizeof(proc_coord), proc_coord);
    MPI_Cart_shift(grid_comm, Y_DIR, 1, &proc_top, &proc_bottom); /* rank of processes proc_top and proc_bottom */
    MPI_Cart_shift(grid_comm, X_DIR, 1, &proc_left, &proc_right);
    if(DEBUG){
        printf("(%i) top %i, right %i, bottom %i, left %i\n",
               proc_rank, proc_top, proc_right, proc_bottom, proc_left);
    }
}
void Setup_Grid()
{
    int x, y, s;
    double source_x, source_y, source_val;
    FILE *f;
    int upper_offset[2];
    Debug("Setup_Subgrid", 0);
    if(proc_rank==0){
        f = fopen("input.dat", "r");
        if (f == NULL)
            Debug("Error opening input.dat", 1);
        fscanf(f, "nx: %i\n", &gridsize[X_DIR]);
        fscanf(f, "ny: %i\n", &gridsize[Y_DIR]);
        fscanf(f, "precision goal: %lf\n", &precision_goal);
        fscanf(f, "max iterations: %i\n", &max_iter);
    }
    MPI_Bcast(&gridsize, 2, MPI_INT, 0, grid_comm);
    MPI_Bcast(&precision_goal, 1, MPI_DOUBLE, 0, grid_comm);
    MPI_Bcast(&max_iter, 1, MPI_INT, 0, grid_comm);

    /* Calculate dimensions of local subgrid */
    /*
    dim[X_DIR] = gridsize[X_DIR] + 2;
    dim[Y_DIR] = gridsize[Y_DIR] + 2;
     */

    offset[X_DIR]=gridsize[X_DIR]*proc_coord[X_DIR]/P_grid[X_DIR];
    offset[Y_DIR]=gridsize[Y_DIR]*proc_coord[Y_DIR]/P_grid[Y_DIR];
    upper_offset[X_DIR]=gridsize[X_DIR]*(proc_coord[X_DIR]+1)/P_grid[X_DIR];
    upper_offset[Y_DIR]=gridsize[Y_DIR]*(proc_coord[Y_DIR]+1)/P_grid[Y_DIR];
    dim[X_DIR]=upper_offset[X_DIR]-offset[X_DIR];
    dim[Y_DIR]=upper_offset[Y_DIR]-offset[Y_DIR];
    dim[X_DIR]+=2;
    dim[Y_DIR]+=2;
    /* allocate memory */
    if ((phi = malloc(dim[X_DIR] * sizeof(*phi))) == NULL)
        Debug("Setup_Subgrid : malloc(phi) failed", 1);
    if ((source = malloc(dim[X_DIR] * sizeof(*source))) == NULL)
        Debug("Setup_Subgrid : malloc(source) failed", 1);
    if ((phi[0] = malloc(dim[Y_DIR] * dim[X_DIR] * sizeof(**phi))) == NULL)
        Debug("Setup_Subgrid : malloc(*phi) failed", 1);
    if ((source[0] = malloc(dim[Y_DIR] * dim[X_DIR] * sizeof(**source))) == NULL)
        Debug("Setup_Subgrid : malloc(*source) failed", 1);
//  printf("size of phi: %llu", sizeof(*phi));
//  printf("size of **phi: %llu", sizeof(**phi));
    for (x = 1; x < dim[X_DIR]; x++)
    {
        phi[x] = phi[0] + x * dim[Y_DIR];
        source[x] = source[0] + x * dim[Y_DIR];
    }

    /* set all values to '0' */
    for (x = 0; x < dim[X_DIR]; x++)
        for (y = 0; y < dim[Y_DIR]; y++)
        {
            phi[x][y] = 0.0;
            source[x][y] = 0;
        }

    /* put sources in field */
    do
    {
        if(proc_rank==0){
            s = fscanf(f, "source: %lf %lf %lf\n", &source_x, &source_y, &source_val);
        }
        MPI_Bcast(&s, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (s==3)
        {
            MPI_Bcast(&source_x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&source_y, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(&source_val, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            x = source_x * gridsize[X_DIR];
            y = source_y * gridsize[Y_DIR];

            x += 1;
            y += 1;
            x=x-offset[X_DIR];
            y=y-offset[Y_DIR];
            if(x>0&&x<dim[X_DIR]-1&&y>0&&y<dim[Y_DIR]-1){
                phi[x][y] = source_val;
                source[x][y] = 1;
            }
            //printf("process_id: %i\n", proc_rank);
        }
    }
    while (s==3);
    if(proc_rank==0){
        fclose(f);
    }

}
void Setup_MPI_Datatypes(){
//    char str1[]="setup mpi datatypes";
//    printf("process %i", proc_rank);
//    printf("%s\n", str1);
    Debug("Setup_MPI_Datatypes", 0);

    MPI_Type_vector(dim[X_DIR] - 2, 1, dim[Y_DIR], MPI_DOUBLE, &border_type[Y_DIR]);

    MPI_Type_commit(&border_type[Y_DIR]);

    /* Datatype for horizontal data exchange (X_DIR) */
    MPI_Type_vector(dim[Y_DIR] - 2, 1, 1, MPI_DOUBLE, &border_type[X_DIR]);

    MPI_Type_commit(&border_type[X_DIR]);

}
void Exchange_Borders(){
    Debug("Exchange_Borders", 0);
    MPI_Sendrecv(&phi[1][1], 1, border_type[Y_DIR], proc_top, 0,
                 &phi[1][dim[Y_DIR]-1], 1, border_type[Y_DIR], proc_bottom, 0,
                 grid_comm, &status);
    MPI_Sendrecv(&phi[1][dim[Y_DIR]-2], 1, border_type[Y_DIR], proc_bottom, 0,
                 &phi[1][0], 1, border_type[Y_DIR], proc_top, 0,
                 grid_comm, &status);
    MPI_Sendrecv(&phi[dim[X_DIR] - 2][1], 1, border_type[X_DIR], proc_right, 0,
                 &phi[0][1], 1, border_type[X_DIR], proc_left, 0,
                 grid_comm, &status);
    MPI_Sendrecv(&phi[1][1], 1, border_type[X_DIR], proc_left, 0,
                 &phi[dim[X_DIR] - 1][1], 1, border_type[X_DIR], proc_right, 0,
                 grid_comm, &status);
    total_data_communicated=total_data_communicated+(dim[X_DIR]-2)*2+(dim[Y_DIR]-2)*2;
}
double Do_Step(int parity)
{
    int x, y;
    double old_phi;
    double c = 0.0;
    double max_err = 0.0;
    int y_calibration=0;
    double OMEGA=1.95;
    /* calculate interior of grid */
    for (x = 1; x < dim[X_DIR] - 1; x++) {
        y_calibration=(x+offset[X_DIR]+parity+1)%2;
        for(y=1+y_calibration;y<dim[Y_DIR]-1;y+=2){
            if (source[x][y] != 1)
            {
                old_phi = phi[x][y];
                phi[x][y]=old_phi+OMEGA*((phi[x + 1][y] + phi[x - 1][y] +
                                          phi[x][y + 1] + phi[x][y - 1]) * 0.25-old_phi);
                if (max_err < fabs(old_phi - phi[x][y]))
                    max_err = fabs(old_phi - phi[x][y]);
            }
        }
    }
//    for(x=1;x<dim[X_DIR]-1;x++){
//        for (y = 1; y < dim[Y_DIR] - 1; y++){
//            if ((x + offset[X_DIR] + y + offset[Y_DIR]) % 2 == parity && source[x][y] != 1){
//                old_phi = phi[x][y];
//                phi[x][y]=old_phi+OMEGA*((phi[x + 1][y] + phi[x - 1][y] +
//                                          phi[x][y + 1] + phi[x][y - 1]) * 0.25-old_phi);
//                if (max_err < fabs(old_phi - phi[x][y]))
//                    max_err = fabs(old_phi - phi[x][y]);
//            }
//
//        }
//    }
    return max_err;
}


void Solve()
{
    int count = 0;
    double delta;
    double delta1, delta2, delta3, delta4, delta5, delta6, delta7, delta8, delta9, delta10;
    double delta_1, delta_2, delta_3, delta_4, delta_5;
    double global_delta;

    Debug("Solve", 0);

    /* give global_delta a higher value then precision_goal */
    global_delta = 2 * precision_goal;
    while (global_delta > precision_goal && count < max_iter)
    {

        //global_parity=0;
        Debug("Do_Step 0", 0);
        delta1 = Do_Step(0);
        border_exchange_start_point=MPI_Wtime();
//        if(proc_rank==0){
//            printf("border_exchange_start_point:%f\n", border_exchange_start_point);
//        }

        Exchange_Borders();
        border_exchange_end_point=MPI_Wtime();
//        if(proc_rank==0){
//            printf("border_exchange_end_point:%f\n", border_exchange_end_point);
//        }

        //printf("process: %i. Execution time for the first exchange_borders(): %f\n", proc_rank, border_exchange_end_point-border_exchange_start_point);
        total_communication_time=total_communication_time+border_exchange_end_point-border_exchange_start_point;
        Debug("Do_Step 1", 0);
        delta2= Do_Step(1);
        border_exchange_start_point=MPI_Wtime();
        Exchange_Borders();
        border_exchange_end_point=MPI_Wtime();
        total_communication_time=total_communication_time+border_exchange_end_point-border_exchange_start_point;
        //printf("process: %i. Execution time for the first exchange_borders(): %f\n", proc_rank, border_exchange_end_point-border_exchange_start_point);
//    delta_1=max(delta1, delta2);
//      Exchange_Borders();
//      delta3= Do_Step(0);
//
//
//    delta4= Do_Step(1);
//
//      //global_parity=1;
//      delta_2=max(delta3, delta4);
//      Exchange_Borders();
//      delta5= Do_Step(0);
//
//      delta6= Do_Step(1);
//
//      delta_3=max(delta5, delta6);
//      Exchange_Borders();
//      delta7= Do_Step(0);
//
//      delta8= Do_Step(1);
//
//      delta_4=max(delta7, delta8);
//      Exchange_Borders();
//      delta8= Do_Step(0);
//
//      delta9= Do_Step(1);
//
//      delta_5=max(delta8, delta9);
//      Exchange_Borders();
//    delta = max(max(delta_1, delta_2), delta_3);
        delta=max(delta1, delta2);
        MPI_Allreduce(&delta, &global_delta, 1, MPI_DOUBLE, MPI_MAX, grid_comm);


        count++;
    }
    printf("The process %i, communication time: %f s\n", proc_rank, total_communication_time);
    //printf("Number of iterations : %i\n", count);
    printf("The process %i. Number of iterations: %i\n", proc_rank, count);
}

void Write_Grid()
{
    int x, y;
    FILE *f;
    char filename[40];
//  printf("process %i", proc_rank);

    sprintf(filename, "output%i.dat", proc_rank);
    puts(filename);
    if ((f = fopen(filename, "w")) == NULL)
        Debug("Write_Grid : fopen failed", 1);

    Debug("Write_Grid", 0);

    for (x = 1; x < dim[X_DIR] - 1; x++)
        for (y = 1; y < dim[Y_DIR] - 1; y++)
            fprintf(f, "%i %i %f\n", x+offset[X_DIR], y+offset[Y_DIR], phi[x][y]);



    fclose(f);
}

void Clean_Up()
{
    Debug("Clean_Up", 0);

    free(phi[0]);
    free(phi);
    free(source[0]);
    free(source);
}

void Merge_files(int P){

    if(proc_rank==0){

        char file_name2[80];
        for(int i=0;i<P;i++) {
            char file_name1[100] = "output";

            sprintf(file_name2, "%d", i);
            char *file_name3 = ".dat";
            char *final_file_name = "final_output.dat";
            int c;
            char buff[255];
            strcat(file_name2, file_name3);
            strcat(file_name1, file_name2);
            FILE *f = fopen(file_name1, "r");
            FILE *final = fopen(final_file_name, "a");
            if(f==NULL){
                puts("could not open files");
                exit(0);
            }
            while(fgets(buff, sizeof(buff), f)!=NULL){
                fputs(buff, final);
            }
            fclose(f);
            fclose(final);
        }
        printf("merge files complete!\n");
    }






}
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    Setup_Proc_Grid(argc, argv);
    //MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    start_timer();
    Setup_Grid();
    Setup_MPI_Datatypes();
    int size_mpi_double;
    Solve();

    Write_Grid();
    MPI_Type_size(MPI_DOUBLE, &size_mpi_double);
    total_data_communicated=total_data_communicated*size_mpi_double;
    printf("total_data_amount_communicated: %llu\n", total_data_communicated);
    print_timer();

    Clean_Up();

    Merge_files(P);
    MPI_Finalize();


    return 0;
}