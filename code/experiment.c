#include <stdio.h>
#include <math.h>

#include "widereach.h"
#include "helper.h"

#define SAMPLE_SEEDS 3
unsigned long int samples_seeds[SAMPLE_SEEDS] = {
    85287339, // 412
    20200621154912, // 378
    20200623170005 // 433
    };

/*#define SAMPLE_SEEDS 30
unsigned long int samples_seeds[SAMPLE_SEEDS] = {
    734517477, 145943044, 869199209, 499223379, 523437323, 964156444,
    248689460, 115706114, 711104006, 311906069, 205328448, 471055100,
    307531192, 543901355, 24851720, 704008414, 2921762, 181094221,
    234474543, 782516264, 519948660, 115033019, 205486123, 657145193,
    83898336, 41744843, 153111583, 318522606, 952537249, 298531860
    };*/

#define MIP_SEEDS 30
unsigned int mip_seeds[MIP_SEEDS] = {
    734517477, 145943044, 869199209, 499223379, 523437323, 964156444,
    248689460, 115706114, 711104006, 311906069, 205328448, 471055100,
    307531192, 543901355, 24851720, 704008414, 2921762, 181094221,
    234474543, 782516264, 519948660, 115033019, 205486123, 657145193,
    83898336, 41744843, 153111583, 318522606, 952537249, 298531860
};

unsigned long int validation_seed = 711104006;

// Compute 10^d, where d is even or d=1, 3
int pow10quick(int d) {
  if (!d) {
    return 1;
  }
  if (d % 2) {
    return 10 * pow10quick(d - 1);
  }
  int partial = pow10quick(d / 2);
  return partial * partial;
}

//#define FACT_MAX 16
#define FACT_MAX 9
unsigned int factorial[FACT_MAX];

void initialize_factorial() {
  factorial[0] = 1.;
  for (size_t i = 1; i < FACT_MAX; i++) {
    factorial[i] = i * factorial[i - 1];
  }
}

double fact(unsigned int n) {
  return (double) factorial[n];
}

double *init_solution(int nmemb, double *solution) {
  for (int i = 1; i < nmemb; i++) {
    solution[i] = .5;
  }
  return solution;
}

int main_glpk() {
    initialize_factorial();
    
    env_t env;
    env.params = params_default();
    /*
     * Simplex: 0.99
     * 
     * breast cancer 0.99
     * red wine 0.04
     * white wine 0.1
     * south german credit  .95 (2, 1)
     * crop mapping  .99 (76, 0.974359); 
     * */
    env.params->theta = 0.9;
    double lambda_factor = 10.;
    // env.params->theta = 0.99;
    env.params->branch_target = 0.0;
    env.params->iheur_method = deep;
    int n = 400;
    // env.params->lambda = 100 * (n + 1); 
    env.params->rnd_trials = 10000;
    // env.params->rnd_trials_cont = 10;
    env.params->rnd_trials_cont = 0;
    
    //size_t dimension = 8;
    size_t dimension = 2;
    
    clusters_info_t clusters[2];
    // int pow10quick(dimension);
    clusters_info_singleton(clusters, n * .8, dimension);
    clusters_info_t *info = clusters + 1;
    info->dimension = dimension;
    size_t cluster_cnt = info->cluster_cnt = 2;
    info->count = CALLOC(cluster_cnt, size_t); 
    info->shift = CALLOC(cluster_cnt, double);
    info->side = CALLOC(cluster_cnt, double);
    info->shift[0] = 0.;
    info->side[0] = 1.;
    info->side[1] = pow(.01, 1. / (double) dimension);
    info->shift[1] = 1. - info->side[1];
    
    info->count[0] = info->count[1] = n / 10;
    
    double side = sqrt(fact(dimension) / fact(FACT_MAX - 1));
    simplex_info_t simplex_info = {
      .count = n,
      .positives = n / 5,
      .cluster_cnt = 1,
      .dimension = dimension,
      .side = side
    };
    
    srand48(validation_seed);
    samples_t *samples_validation;
    // samples_validation = random_simplex_samples(&simplex_info);
    samples_validation = random_sample_clusters(clusters);
    //FILE *infile;
    //infile =
       // fopen("../../data/breast-cancer/wdbc-validation.dat", "r");
       // fopen("../../data/wine-quality/winequality-red-validation.dat", "r");
       // fopen("../../data/wine-quality/red-cross/winequality-red-2-validation.dat", "r");
       // fopen("../../data/wine-quality/winequality-white-validation.dat", "r");
       // fopen("../../data/wine-quality/white-cross/winequality-white-2-validation.dat", "r");
       // fopen("../../data/south-german-credit/SouthGermanCredit-validation.dat", "r");
       // fopen("../../data/south-german-credit/cross/SouthGermanCredit-2-validation.dat", "r");
       // fopen("../../data/crops/small-sample-validation.dat", "r");
       // fopen("../../data/crops/cross/small-sample-2-validation.dat", "r");
    //  fopen("./sample.dat", "r");
    //samples_validation = read_binary_samples(infile);
    //fclose(infile);
    /* glp_printf("Validation\n");
    print_samples(samples_validation);
    return 0; */
    
    double *h;
    dimension = samples_validation->dimension;
    int solution_size = dimension + samples_total(samples_validation) + 3;
    double *solution = CALLOC(solution_size, double);
    
    // for (int s = 0; s < SAMPLE_SEEDS; s++) {
    for (int s = 0; s < 1; s++) {
        srand48(samples_seeds[s]);
	glp_printf("Found norm 0 in feature scaling\n");

        glp_printf("Sample seed: %lu\n", samples_seeds[s]);
	glp_printf("Found norm 0 in feature scaling 12\n");
    
        samples_t *samples;
        
        // samples = random_samples(n, n / 2, dimension);
        samples = random_sample_clusters(clusters);
        // samples = random_simplex_samples(&simplex_info);
        // infile = fopen("../../data/breast-cancer/wdbc-training.dat", "r");
            // fopen("../../data/wine-quality/winequality-red-training.dat", "r");
            // fopen("../../data/wine-quality/red-cross/winequality-red-2-training.dat", "r");
            // fopen("../../data/wine-quality/winequality-white-training.dat", "r"); 
            // fopen("../../data/wine-quality/white-cross/winequality-white-2-training.dat", "r");
            // fopen("../../data/south-german-credit/SouthGermanCredit-training.dat", "r");
            // fopen("../../data/south-german-credit/cross/SouthGermanCredit-2-training.dat", "r");
            // fopen("../../data/cross-sell/train-nocat.dat", "r"); */
            // fopen("../../data/crops/sample.dat", "r");
	    // fopen("../../data/crops/small-sample-training.dat", "r");
            // fopen("../../data/crops/cross/small-sample-2-training.dat", "r");
	//fopen("./small-sample.dat", "r");
        //samples = read_binary_samples(infile);
        //fclose(infile);
        
        env.samples = samples;
        n = samples_total(samples);
        env.params->lambda = lambda_factor * (n + 1);
        
        /* print_samples(env.samples);
        return 0; */  
        
        // for (int t = 0; t < MIP_SEEDS; t++) {    
        for (int t = 0; t < 1; t++) {
        // if (0) { int t=0;
        // for (int t = 0; t < 6; t++) {
            unsigned int *seed = mip_seeds + t;
            // precision_threshold(seed, &env); See branch theta-search
            // precision_scan(seed, &env);
            // glp_printf("Theta: %g\n", env.params->theta);
            h = single_run(seed, 120000, &env);
            hyperplane_to_solution_parts(h + 1, 
                                         init_solution(solution_size, solution), 
                                         env.params, 
                                         samples_validation); 
            glp_printf("Validation: %u\t%lg\n", 
                       reach(solution, samples_validation),
                       precision(solution, samples_validation)); 
            free(h);
        }
        free(delete_samples(samples));
    }
    
    free(solution);
    free(delete_samples(samples_validation));
    delete_clusters_info(clusters);
    delete_clusters_info(clusters + 1);
    free(env.params);

    return 0;
}

void feature_scaling(env_t *env) {
  size_t dim = env->samples->dimension;
  int nsamples = 0;
  for(int c = 0; c < env->samples->class_cnt; c++)
    nsamples += env->samples->count[c];
  double *norms = CALLOC(nsamples, double);
  for(int c = 0; c < env->samples->class_cnt; c++) {
    size_t cnt = env->samples->count[c];
    for(int d = 0; d < dim; d++) {
      //selected class c, feature d
      for(int i = 0; i < cnt; i++)
	norms[d] += fabs(env->samples->samples[c][i][d]);
    }
  }
  //now that norms have been computed, we go back and scale each feature
  for(int c = 0; c < env->samples->class_cnt; c++) {
    size_t cnt = env->samples->count[c];
    for(int d = 0; d < dim; d++) {
      if(norms[d] == 0){
	printf("Found norm 0 in feature scaling\n");
	continue;
      }
                  
      for(int i = 0; i < cnt; i++)
	env->samples->samples[c][i][d] /= norms[d];
    }
  }
}

void write_samples(samples_t *samples, char *path) {
  FILE *f = fopen(path, "w");
  fprintf(f, "%lu %lu %lu\n", samples->dimension, samples->count[0], samples->count[1]);
  for(size_t class = 0; class < samples->class_cnt; class++) {
    for(size_t i = 0; i < samples->count[class]; i++) {
      for(int j = 0; j < samples->dimension; j++) {
	fprintf(f, "%g ", samples->samples[class][i][j]);
      }
      fprintf(f, "\n");
    }
  }
}


typedef struct exp_res_t {
  double reach;
  double prec;
} exp_res_t;

exp_res_t experiment(int param_setting) {
    initialize_factorial();
    
    env_t env;
    env.params = params_default();
    /*
     * Simplex: 0.99
     * 
     * breast cancer 0.99
     * red wine 0.04
     * white wine 0.1
     * south german credit  .9 (2, 1) (originally said 0.95 here)
     * crop mapping  .99 (76, 0.974359); 
     * */
    env.params->theta = 0.9;
    double lambda_factor = 10;
    env.params->branch_target = 0.0;
    env.params->iheur_method = deep;
    int n = 40;
    // env.params->lambda = 100 * (n + 1); 
    env.params->rnd_trials = 10000;
    // env.params->rnd_trials_cont = 10;
    env.params->rnd_trials_cont = 0;
    
    //size_t dimension = param_setting;
    size_t dimension = 2;
    
    clusters_info_t clusters[2];
    // int n = pow10quick(dimension);
    clusters_info_singleton(clusters, n * .8, dimension);
    clusters_info_t *info = clusters + 1;
    info->dimension = dimension;
    size_t cluster_cnt = info->cluster_cnt = 2;
    info->count = CALLOC(cluster_cnt, size_t); 
    info->shift = CALLOC(cluster_cnt, double);
    info->side = CALLOC(cluster_cnt, double);
    info->shift[0] = 0.;
    info->side[0] = 1.;
    info->side[1] = pow(.01, 1. / (double) dimension);
    info->shift[1] = 1. - info->side[1];
    
    info->count[0] = info->count[1] = n / 10;
    
    double side = sqrt(fact(dimension) / fact(FACT_MAX - 1));
    size_t cluster_sizes[] = {n/40, n/80, n/80, n/20};
    simplex_info_t simplex_info = {
      .count = n,
      .positives = n / 5,
      .cluster_cnt = 1,
      .dimension = dimension,
      .side = side,
      //.cluster_sizes = cluster_sizes
    };
    /*for(int i = 0; i < simplex_info.cluster_cnt; i++)
      printf("%s%ld", i == 0 ? "cluster sizes: " : ", ", cluster_sizes[i]);
      printf("\n");*/
    srand48(validation_seed);
    samples_t *samples_validation;
    samples_validation = random_samples(n, n / 2, dimension);
    //samples_validation = random_sample_clusters(clusters);
    //samples_validation = random_simplex_samples(&simplex_info);
    FILE *infile;
    infile =
      //fopen("../../data/breast-cancer/wdbc-validation.dat", "r");
      //fopen("../../data/wine-quality/winequality-red-validation.dat", "r");
      //fopen("../../data/wine-quality/red-cross/winequality-red-2-validation.dat", "r");
      //fopen("../../data/wine-quality/winequality-white-validation.dat", "r");
      //fopen("../../data/wine-quality/white-cross/winequality-white-2-validation.dat", "r");
      fopen("../../data/south-german-credit/SouthGermanCredit-validation.dat", "r");
      //fopen("../../data/south-german-credit/cross/SouthGermanCredit-2-validation.dat", "r");
      //fopen("../../data/crops/small-sample-validation.dat", "r");
      //fopen("../../data/crops/cross/small-sample-2-validation.dat", "r");
      //fopen("../../data/finance_data/finance-valid.dat", "r");
       // fopen("./sample.dat", "r");
    //samples_validation = read_binary_samples(infile);
    fclose(infile);
    /* glp_printf("Validation\n");
    print_samples(samples_validation);
    return 0; */

    //print_samples(samples_validation);

    double *h;
    dimension = samples_validation->dimension;
    int solution_size = dimension + samples_total(samples_validation) + 3;
    double *solution = CALLOC(solution_size, double);

    int ntests = SAMPLE_SEEDS*MIP_SEEDS;

    unsigned int *reaches = CALLOC(ntests, unsigned int);
    double *precisions = CALLOC(ntests, double);
    int k = 0;
    
    for (int s = 0; s < SAMPLE_SEEDS; s++) {
    //for (int s = 0; s < 1; s++) {
        srand48(samples_seeds[s]);
	printf("Found norm 0 in feature scaling\n");


        printf("Sample seed: %lu\n", samples_seeds[s]);
	printf("Found norm 0 in feature scaling\n");

    
        samples_t *samples;
        //samples = random_samples(n, n / 2, dimension);
        //samples = random_sample_clusters(clusters);
	//samples = random_simplex_samples(&simplex_info);
        infile =
	  //fopen("../../data/breast-cancer/wdbc-training.dat", "r");
	  //fopen("../../data/wine-quality/winequality-red-training.dat", "r");
	  //fopen("../../data/wine-quality/red-cross/winequality-red-2-training.dat", "r");
	  //fopen("../../data/wine-quality/winequality-white-training.dat", "r"); 
          //fopen("../../data/wine-quality/white-cross/winequality-white-2-training.dat", "r");
          //fopen("../../data/south-german-credit/SouthGermanCredit-training.dat", "r");
	  //fopen("../../data/south-german-credit/cross/SouthGermanCredit-2-training.dat", "r");
            // fopen("../../data/cross-sell/train-nocat.dat", "r"); 
            // fopen("../../data/crops/sample.dat", "r");
	  //fopen("../../data/crops/small-sample-training.dat", "r");
            // fopen("../../data/crops/cross/small-sample-2-training.dat", "r");
	  //fopen("../../data/finance_data/finance-train.dat", "r");
	    // fopen("./small-sample.dat", "r");
	  //full data sets:
	  //fopen("../../data/breast-cancer/wdbc.dat", "r");
	  //fopen("../../data/wine-quality/red-cross/winequality-red.dat", "r");
	  //fopen("../../data/wine-quality/white-cross/winequality-white-1.dat", "r");
	  fopen("../../data/south-german-credit/SouthGermanCredit.dat", "r");
	  //fopen("../../data/crops/small-sample.dat", "r");
    // fopen("../../data/south-german-credit/SouthGermanCredit.dat", "r");

  
	samples = read_binary_samples(infile);
	fclose(infile);
	//write_samples(samples, "2cluster4000.dat");
	//exit(0);

	//print_samples(samples);

        env.samples = samples;
	add_bias(samples);
	feature_scaling(&env);
	normalize_samples(samples);
	//print_samples(samples);
        n = samples_total(samples);
        env.params->lambda = lambda_factor * (n + 1);
	env.params->epsilon_precision = 3./990;
	//env.params->epsilon_precision = 3000./990000;

        //print_samples(env.samples);
        //return (exp_res_t) {0, 0};
	
	//feature_scaling(&env);
        
        for (int t = 0; t < MIP_SEEDS; t++) {
	//for (int t = 0; t < 1; t++) {
        // if (0) { int t=0;
        // for (int t = 0; t < 6; t++) {
            unsigned int *seed = mip_seeds + t;
	    printf("s = %d, t = %d\n", s, t);
            // precision_threshold(seed, &env); See branch theta-search
            // precision_scan(seed, &env);
            // glp_printf("Theta: %g\n", env.params->theta);

	    /*h = gurobi_relax(seed, 120000, 1200, &env);
	    double *soln = blank_solution(samples);
	    double obj = hyperplane_to_solution(h, soln, &env);
	    printf("Reach = %d\n", reach(soln, env.samples));
	    printf("Precision = %g\n", precision(soln, env.samples));
	    int npos = 0, nneg = 0;
	    for(int i = 0; i < n; i++)
	      if(soln[i] == 1) npos++;
	      else if(soln[i] == 0) nneg++;
	      printf("%d pos, %d neg\n", npos, nneg);
	    exit(0);*/
	    
	    /*h = single_siman_cones_run(seed, 0, &env, NULL);
	      exit(0);*/
	    /*for(int i = 0; i <= 2; i++) {
	      double *insep = compute_inseparabilities(&env, i);
	      printf("i = %d => viol = %g\n", i, *insep);
	      free(insep);
	    }
	    exit(0);*/

	    //double *h = single_gurobi_cones_run(seed, 120000, 1200, &env);
	    //Training results testing:
	    if(param_setting <= 0) {
	      double *h = single_exact_run(&env, 100);
	      double obj = hyperplane_to_solution(h, NULL, &env);
	      printf("Solved. Obj = %g\n", obj);
	      printf("Hyperplane: ");
	      for(int i = 0; i < env.samples->dimension; i++)
		printf("%g%s", h[i], (i == env.samples->dimension - 1) ? "\n" : " ");
	      exit(0);
	    } else if(param_setting == 1) {
	      //use gurobi
	      gurobi_param p = {param_setting, 0, 0, GRB_INFINITY, -1, 0.15, -1};
	      h = single_gurobi_run(seed, 120000, 1200, &env, &p);
	      //h = single_siman_run(seed, 0, &env, h+1);
	      printf("Objective = %0.3f\n", h[0]);
	    } else if (param_setting == 2) {
	      //use glpk
	      h = single_run(seed, 120000, &env);
	      printf("Objective = %0.3f\n", h[0]);
	    } else {
	      //best random hyperplane
	      srand48(*seed);
	      h = best_random_hyperplane(1, &env);
	      double *random_solution = blank_solution(samples);
	      double random_objective_value = hyperplane_to_solution(h, random_solution, &env);
	      printf("Initial reach = %0.3f\n", random_objective_value);
	      printf("Hyperplane: ");
	      for(int i = 0; i <= dimension; i++)
		printf("%0.5f%s", h[i], (i == dimension) ? "\n" : " ");
	      free(random_solution);
	    }

	    printf("Hyperplane: ");
	    for(int i = 1; i <= env.samples->dimension + 1; i++)
	      printf("%0.5f%s", h[i], (i == env.samples->dimension + 1) ? "\n" : " ");
	    //printf("Objective value: %0.3f\n", h[0]);

	    reaches[k] = reach(h, env.samples);
	    precisions[k] = precision(h, env.samples);
	    printf("P = %0.3f\n", precisions[k]);
	    if(isnan(precisions[k])) precisions[k] = 0;
	   
	    printf("Training: %u\t%lg\n", 
		   reaches[k],
		   precisions[k]);

	    printf("Validation: %u\t%lg\n", reach(h, samples_validation), precision(h, samples_validation));

	    exit(0);
	    
	    k++;
            free(h);
	}
        free(delete_samples(samples));
    }
    free(solution);
    free(delete_samples(samples_validation));
    delete_clusters_info(clusters);
    delete_clusters_info(clusters + 1);
    free(env.params);

    printf("Reaches: ");
    for(int i = 0; i < ntests; i++)
      printf("%d, ", reaches[i]);
    printf("\n");

    printf("Precisions: ");
    for(int i = 0; i < ntests; i++)
      printf("%0.3f, ", precisions[i]);
    printf("\n");

    printf("n = %d. Cluster size = %0.3f\n", n, ((double) n)/10);
    
    int tot_reach = 0;
    double tot_prec = 0;
    for(int i = 0; i < ntests; i++) {
      tot_reach += reaches[i];
      tot_prec += precisions[i];
    }
    double avg_reach = ((double) tot_reach) / ntests;
    double avg_prec = tot_prec / ntests;

    //printf("Average reach = %0.3f, average precision = %0.3f\n", avg_reach, avg_prec);
    printf("Average reach/cluster size = %0.3f, average precision = %0.3f\n", avg_reach/(((double) n)/20), avg_prec);
    free(reaches);
    free(precisions);

    exp_res_t res;
    res.prec = avg_prec;
    res.reach = avg_reach;
    return res;
}

int main(int argc, char *argv[]) {
	printf("Found norm 0 in feature scaling\n");

  /*int nsettings = 10;
  exp_res_t *results = CALLOC(nsettings, exp_res_t);
  for(int i = 0; i < nsettings; i++)
    results[i] = experiment(i);


  printf("Setting | Average Reach | Average Precision\n");
  printf("--------|---------------|------------------\n");
  for(int i = 0; i < nsettings; i++)
    printf("  %d    |      %d       |    %0.5f\n", i, results[i].reach, results[i].prec);

    free(results);*/
  
  exp_res_t res;
  
  if(argc == 1)
    res = experiment(0);
  else
    res = experiment(atoi(argv[1]));

  printf("RESULTS: %0.5f, %0.5f\n", res.prec, res.reach);
  
  return 0;
}
