#include <stdio.h>
#include <math.h>

#include "widereach.h"
#include "helper.h"

#include <gsl/gsl_rng.h> //can remove these after testing gpu
#include <gsl/gsl_randist.h>
#include <memory.h>

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

unsigned long int validation_seed = 593587157;

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
	printf("Found norm 0 on dimension %d\n", d);
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

//TODO: move this somewhere else
double sample_angle(samples_t *samples, sample_locator_t li, sample_locator_t lj) {
  double dot = 0, normi = 0, normj = 0;
  size_t d = samples->dimension;
  for(int k = 0; k < d; k++) {
    double elti = samples->samples[li.class][li.index][k];
    double eltj = samples->samples[lj.class][lj.index][k];
    dot += elti*eltj;
    normi += elti*elti;
    normj += eltj*eltj;
  }
  normi = sqrt(normi);
  normj = sqrt(normj);
  return acos(dot/(normi*normj));
}

double min_angle(samples_t *samples) {
  double min = 1e101;
  for(int classi = 0; classi < samples->class_cnt; classi++) {
    for(int i = 0; i < samples->count[classi]; i++) {
      for(int classj = classi; classj < samples->class_cnt; classj++) {
	int startj = (classj == classi) ? i + 1 : 0;
	for(int j = startj; j < samples->count[classj]; j++) {
	  sample_locator_t li = {.class = classi, .index = i};
	  sample_locator_t lj = {.class = classj, .index = j};
	  double ang = sample_angle(samples, li, lj);
	  if(ang == 0) {
	    printf("li: {%d, %d}, lj: {%d, %d}\n", classi, i, classj, j);
	    printf("i: ");
	    for(int k = 0; k < samples->dimension; k++) {
	      printf("%g, ", samples->samples[classi][i][k]);
	    }
	    printf("\nj: ");
	    for(int k = 0; k < samples->dimension; k++) {
	      printf("%g, ", samples->samples[classj][j][k]);
	    }
	    printf("\n");
	    exit(0);
	  }
	  if(ang < min) min = ang;
	}
      }
    }
  }
  return min;
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
    env.params->theta = 0.1;
    double lambda_factor = 10;
    env.params->branch_target = 0.0;
    env.params->iheur_method = simple;
    int n = 1000;
    // env.params->lambda = 100 * (n + 1); 
    env.params->rnd_trials = 10000;
    // env.params->rnd_trials_cont = 10;
    env.params->rnd_trials_cont = 0;
    
    //size_t dimension = param_setting;
    size_t dimension = 8;
    
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
      .scale = 1
    };
    /*for(int i = 0; i < simplex_info.cluster_cnt; i++)
      printf("%s%ld", i == 0 ? "cluster sizes: " : ", ", cluster_sizes[i]);
      printf("\n");*/
    srand48(validation_seed);
    samples_t *samples_validation;
    //samples_validation = random_samples(n, n / 2, dimension);
    //samples_validation = random_sample_clusters(clusters);
    //samples_validation = random_simplex_samples(&simplex_info);
    FILE *infile;
    infile =
      //fopen("../../data/breast-cancer/wdbc-validation.dat", "r");
      //fopen("../../data/wine-quality/winequality-red-validation.dat", "r");
      //fopen("../../data/wine-quality/red-cross/winequality-red-2-validation.dat", "r");
      fopen("../../data/wine-quality/winequality-white-validation.dat", "r");
      //fopen("../../data/wine-quality/white-cross/winequality-white-2-validation.dat", "r");
      //fopen("../../data/south-german-credit/SouthGermanCredit-validation.dat", "r");
      //fopen("../../data/south-german-credit/cross/SouthGermanCredit-2-validation.dat", "r");
      //fopen("../../data/crops/small-sample-validation.dat", "r");
      //fopen("../../data/crops/cross/small-sample-2-validation.dat", "r");
      //fopen("../../data/finance_data/finance-valid.dat", "r");
       // fopen("./sample.dat", "r");
    samples_validation = read_binary_samples(infile);
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
        printf("Sample seed: %lu\n", samples_seeds[s]);
    
        samples_t *samples;
	//samples = random_samples(n, n / 2, dimension);
        //samples = random_sample_clusters(clusters);
	//samples = random_simplex_samples(&simplex_info);
        infile =
	  //fopen("../../data/breast-cancer/wdbc-training.dat", "r");
	  //fopen("../../data/wine-quality/winequality-red-training.dat", "r");
	  //fopen("../../data/wine-quality/red-cross/winequality-red-2-training.dat", "r");
	  fopen("../../data/wine-quality/winequality-white-training.dat", "r"); 
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
	  //fopen("../../data/south-german-credit/SouthGermanCredit.dat", "r");
	  //fopen("../../data/crops/small-sample.dat", "r");
	  //PCA (d=5):
	  //fopen("../../data/breast-cancer/wdbc.dat", "r");
	  //fopen("../../data/wine-quality/red-cross/winequality-red.dat", "r");
	  //fopen("../../data/wine-quality/white-cross/winequality-white-1_pca5.dat", "r");
	  //fopen("../../data/wine-quality/white-cross/winequality-white-1_pca10.dat", "r");
	  //fopen("../../data/south-german-credit/SouthGermanCredit_pca5.dat", "r");
	  //fopen("../../data/south-german-credit/SouthGermanCredit_pca5_affine.dat", "r");
	  //fopen("../../data/crops/small-sample_pca5.dat", "r");
	  //generated datasets:
	  //fopen("../../instance_generation/instance_1.dat", "r");
	  //fopen("../../instance_generation/instance_1_pca15.dat", "r");
	  //fopen("../../instance_generation/instance_3.dat", "r");
	  //fopen("../../instance_generation/instance_3_pca15.dat", "r");
	  //SGC encodings:
	  //fopen("../../data/south-german-credit/SGC_full.dat", "r");
	  //crops classes:
	  //fopen("../../data/crops/small-sample-broadleaf.dat", "r");
	  //fopen("../../data/crops/small-sample-canola.dat", "r");
	  //fopen("../../data/crops/small-sample-corn.dat", "r");
	  //fopen("../../data/crops/small-sample-oat.dat", "r");
	  //fopen("../../data/crops/small-sample-pea.dat", "r");
	  //fopen("../../data/crops/small-sample-soy.dat", "r");
	  //fopen("../../data/crops/small-sample-wheat.dat", "r");
	  //new datasets:
	  //fopen("../../data/rice/rice.dat", "r");
	  //fopen("../../data/crops/small-sample-corn.dat", "r");
	  //fopen("../../data/wine-quality/white-cross/winequality-white-1-dedup.dat", "r");
	  //fopen("../../data/glass/glass.dat", "r");
	  //fopen("../../data/lympho/lympho.dat", "r");
	  //fopen("../../data/vowels/vowels.dat", "r");
	  //fopen("../../data/thyroid/thyroid.dat", "r");
	  //SMOTE results:
	  //fopen("../../data/wine-quality/white-cross/winequality-white-1_smote_0.5.dat", "r");
	  //fopen("../../data/wine-quality/white-cross/winequality-white-1_smote_0.3.dat", "r");
	
	samples = read_binary_samples(infile);
	fclose(infile);
	//write_samples(samples, "2cluster4000.dat");

	//print_samples(samples);
	env.samples = samples;

	/*add_bias(samples);
	printf("eps = %g\n", eps_sample(&env, 1000));
	exit(0);*/


	//add_bias(samples);
	//feature_scaling(&env);
	//normalize_samples(samples);
	//env.samples = samples;
	//	print_samples(samples);
        n = samples_total(samples);
        env.params->lambda = lambda_factor * (n + 1);
	env.params->epsilon_precision = 3./990;
	//env.params->epsilon_precision = 3000./990000;

	/*printf("normalized:\n");
	print_samples(env.samples);
	exit(0);*/

        //print_samples(env.samples);
        //return (exp_res_t) {0, 0};
	
	//feature_scaling(&env);
	/*env.params->epsilon_positive = 1e-5;
	  env.params->epsilon_negative = 1e-5;*/
        
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

	    /*printf("insep = %g\n", insep_score(&env));
	      exit(0);*/

	    //double *h = single_gurobi_cones_run(seed, 120000, 1200, &env);

	    //testing cuda:
	    /*store_samples_cuda(samples);
	    printf("stored\n");
	    vsamples_t *vs = samples_to_vec(samples);
	    env.vsamples = vs;
	    size_t d = vs->dimension;
	    int N = 1000;
	    gsl_rng *rng = gsl_rng_alloc(gsl_rng_taus);
	    gsl_rng_set(rng, rand());
	    for(int i = 0; i < N; i++)  {
	      double *h = CALLOC(d, double);
	      gsl_ran_dir_nd(rng, d, h);
	      gsl_vector w = gsl_vector_view_array(h, d).vector;
	      class_res res = classify_cuda(&env, &w);
	      free_class_res(&res);
	      free(h);
	    }
	    printf("done\n");
	    exit(0);*/
    
	    /*gurobi_param p1 = {
	      .threads = 16,
	      .MIPFocus = 0,
	      .ImproveStartGap = 0,
	      .ImproveStartTime = GRB_INFINITY,
	      .VarBranch = -1,
	      .Heuristics = 0.05,
	      .Cuts = -1,
	      .RINS = -1,
	      .method = 3,
	      .init = NULL
	    };
	    env.params->epsilon_positive *= 2;
	    env.params->epsilon_negative *= 2;
	    env.params->epsilon_precision *= 2;
	    //env.params->rnd_trials = 5;
	    //h = single_gurobi_run(seed, 6000, 1200, &env, &p1);
	    h = best_random_hyperplane_unbiased(1, &env);
	    env.params->epsilon_positive /= 2;
	    env.params->epsilon_negative /= 2;
	    env.params->epsilon_precision /= 2;
	    double *soln2 = blank_solution(samples);
	    double obj2 = hyperplane_to_solution(h, soln2, &env);
	    printf("Objective = %0.3f\n", obj2);

	    printf("reach = %d\n", reach(soln2, env.samples));
	    printf("prec = %g\n", precision(soln2, env.samples));
	    printf("Hyperplane: ");
	    for(int i = 0; i <= samples->dimension+1; i++)
	      printf("%g%s", h[i], (i == samples->dimension+1) ? "\n" : " ");


	    int *cone = expand_cone(&env, h);
	    //int *cone = expand_cone(&env, h+1);
	    for(int i = 0; i < samples_total(samples); i++) {
	      printf("i = %4d | %d | %d\n", i, (int) soln2[i+env.samples->dimension+2], cone[i]);
	    }
	    //cone[0] = 1; //for testing (forcing reach > 0)
	    gurobi_param p = {
	      .threads = 0,
	      .MIPFocus = 0,
	      .ImproveStartGap = 0,
	      .ImproveStartTime = GRB_INFINITY,
	      .VarBranch = -1,
	      .Heuristics = 0.05,
	      .Cuts = -1,
	      .RINS = -1,
	      .method = 6,
	      .init = h, //soln2+1,
	      .cone = cone
	    };
	    h = single_gurobi_run(seed, 120000, 1200, &env, &p);
	    printf("Objective = %0.3f\n", h[0]);
	    printf("Hyperplane: ");
	    for(int i = 0; i <= samples->dimension+1; i++)
	      printf("%g%s", h[i], (i == samples->dimension+1) ? "\n" : " ");

	    env.params->epsilon_positive = 0;
	    env.params->epsilon_negative = 0;
	    env.params->epsilon_precision = 0;
	    double *soln1 = blank_solution(samples);
	    double obj1 = hyperplane_to_solution(h+1, soln1, &env);
	    printf("Objective = %0.3f\n", obj1);

	    printf("reach = %d\n", reach(soln1, env.samples));
	    printf("prec = %g\n", precision(soln1, env.samples));
	    printf("Hyperplane: ");
	    for(int i = 1; i <= env.samples->dimension + 1; i++)
	      printf("%0.5f%s", soln1[i], (i == env.samples->dimension + 1) ? "\n" : " ");
	    exit(0);*/
	    /*add_bias(samples);
	    normalize_samples(samples);
	    printf("minimal angle = %g\n", min_angle(samples));
	    //printf("cos = %g\n", cos(min_angle(samples)));
	    exit(0);*/

	    //Training results testing:
	    if(param_setting <= 0) {
	      //double *h = single_exact_run(&env, 900);
	      add_bias(samples);
	      add_bias(samples_validation);
	      //feature_scaling(&env);
	      //normalize_samples(samples);
	      //print_samples(samples);

	      env.params->greer_params = (struct greer_params) {
		.method = 2,
		.use_heapq = 0,
		.trunc = 0,
		.trim = 0,
		.max_heapq_size = -1,
		.mcts_ucb_const = 100,
		.beam_width = 100,
		.classify_cuda = 0,
		.obj_code = WRC,
		.no_displace = 0,
		.bnb = 0,
	      };

	      //double v0[] = {-0.000648874, -0.00520848, 0.0440311, 0.000688538, -0.171792, -0.000330577, 9.81869e-05, -0.777746, 0.0391895, -0.0820617, 0.00630367, 0.59609};
	      h = best_random_hyperplane_unbiased(1, &env);
	      //h = CALLOC(dimension, double); h[0] = 1;
	      //h = best_random_hyperplane_projection(1, &env);

	      
	      
	      /*h = single_greer_run(&env, h);
	      env.params->greer_params.beam_width = 1000;
	      env.params->greer_params.obj_code = WRC;*/
	      h = single_greer_run(&env, h);

	      /*double obj = hyperplane_to_solution(h, NULL, &env);
	      printf("Solved. Obj = %g\n", obj);
	      printf("Hyperplane: ");
	      for(int i = 0; i < env.samples->dimension; i++)
	      printf("%g%s", h[i], (i == env.samples->dimension - 1) ? "\n" : " ");*/
	      //free(delete_samples(samples));
	      //free(h);
	      //exit(0);

	    } else if(param_setting == 1) {
	      //use gurobi
	      /*double E = 1e-3;
	      double R = 3./990;
	      env.params->epsilon_positive = E;
	      env.params->epsilon_negative = E;
	      env.params->epsilon_precision = E*positives(samples)*(1+1/env.params->theta)+(1+E)*R;*/
	      
	      /*add_bias(samples);
	      normalize_samples(env.samples);
	      add_bias(samples_validation);
	      normalize_samples(samples_validation);*/

	      //h = best_random_hyperplane_unbiased(1, &env);
	      /*gurobi_param p1 = {
		.threads = 0,
		.MIPFocus = 0,
		.ImproveStartGap = 0,
		.ImproveStartTime = GRB_INFINITY,
		.VarBranch = 0,
		.Heuristics = 0.05,
		.Cuts = -1,
		.RINS = -1,
		.method = 0,
		.init = NULL
	      };

	      h = single_gurobi_run(seed, 120000, 1200, &env, &p1);

	      env.params->epsilon_positive = env.params->epsilon_negative = 0;
	      double *soln = blank_solution(samples);
	      double obj = hyperplane_to_solution(h+1, soln, &env);
	    
	      printf("obj %g, prec %g, reach %d\n", obj, precision(soln, samples), reach(soln, samples));
	      exit(0);*/

	      gurobi_param p = {
		.threads = 0,
		.MIPFocus = 3,
		.ImproveStartGap = 0,
		.ImproveStartTime = GRB_INFINITY,
		.VarBranch = 0,
		.Heuristics = 0.05,
		.Cuts = -1,
		.RINS = -1,
		.method = 7,
		.init = NULL
	      };

	      //double epss[11] = {1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1};
	      double thetas[7] = {0.05, 0.15, 0.2, 0.3, 0.5, 0.7, 0.9};
	      double lfs[7] = {20, 20, 5, 10, 2, 10, 10};
	      double objs[11];
	      double reaches[11];
	      double precisions[11];
	      double v_reaches[11];
	      double v_precisions[11];
	      time_t times[11];
	      int tm_lim = 400000; //10 minutes
	      double epsp = env.params->epsilon_negative;
	      double epsn = env.params->epsilon_negative;
	      for(int i = 0; i < 7; i++) {
		//env.params->epsilon_positive = env.params->epsilon_negative = epss[i];
		//env.params->epsilon_positive = epss[i];
		env.params->epsilon_negative = epsn;
		env.params->epsilon_positive = epsp;
		env.params->theta = thetas[i];
		env.params->lambda = lfs[i] * (n + 1);
		time_t start = time(0);
		h = single_gurobi_run(seed, tm_lim, 1200, &env, &p);
		if(!h) {
		  objs[i] = reaches[i] = precisions[i] = times[i] = -1;
		  continue;
		}
		time_t end = time(0);
		env.params->epsilon_positive = env.params->epsilon_negative = 0;

		double *soln = blank_solution(samples);
		double obj = hyperplane_to_solution(h+1, soln, &env);

		objs[i] = h[0];
		reaches[i] = reach(soln, env.samples);
		precisions[i] = precision(soln, env.samples);

		double *v_soln = blank_solution(samples_validation);
		env.samples = samples_validation;
		hyperplane_to_solution(h+1, v_soln, &env);
		env.samples = samples;
		v_reaches[i] = reach(soln, samples_validation);
		v_precisions[i] = precision(soln, samples_validation);
		times[i] = end-start;
		free(h);
		free(soln);
		free(v_soln);
	      }
	      /*printf("eps    |  time  |  obj  | t_reach | t_prec | v_reach | v_prec \n");
	      printf("--------------------------------------------------------------\n");
	      for(int i = 0; i < 11; i++) {
		printf("%6g | %6ld | %5g | %7g | %6g | %7g | %6g\n", epss[i], times[i], objs[i], reaches[i], precisions[i], v_reaches[i], v_precisions[i]);
		}*/
	      printf("theta  |  time  |  obj  | t_reach | t_prec | v_reach | v_prec \n");
	      printf("--------------------------------------------------------------\n");
	      for(int i = 0; i < 7; i++) {
		printf("%6g | %6ld | %5g | %7g | %6g | %7g | %6g\n", thetas[i], times[i], objs[i], reaches[i], precisions[i], v_reaches[i], v_precisions[i]);
	      }
	      exit(0);
	      //env.params->rnd_trials = 5;
	      //env.params->epsilon_positive = 1e-5;
	      //env.params->epsilon_negative = 1e-5;
	      env.params->epsilon_positive *= 2;
	      env.params->epsilon_negative *= 2;
	      h = single_gurobi_run(seed, 120000, 1200, &env, &p);
	      //env.params->epsilon_positive /= 2;
	      //env.params->epsilon_negative /= 2;
	      //h = single_gurobi_run(seed, 120000, 1200, &env, NULL);
	      //h = gurobi_relax(seed, 120000, 1200, &env);
	      //h = single_gurobi_run(seed, 12000000, 1200, &env, &p);
	      //h = single_siman_run(seed, 0, &env, h+1);
	      printf("Objective = %0.3f\n", h[0]);
	      printf("Hyperplane: ");
	      for(int i = 0; i <= samples->dimension+1; i++)
		printf("%g%s", h[i], (i == samples->dimension+1) ? "\n" : " ");

	      h++; //so that making the solution below works (small memory leak)
	      //p.method = 5;
	      //p.init = h+1;
	      /*env.params->epsilon_positive = 0;
	      env.params->epsilon_negative = 0;
	      env.params->epsilon_precision = 0;*/

	      //h = single_gurobi_run(seed, 120000, 1200, &env, &p);
	      /*printf("Result: ");
	      for(int i = 0; i <= samples->dimension+1+samples_total(samples); i++)
		printf("%g ", h[i]);
		printf("\n");*/

	      
	      /*double *random_solution = blank_solution(samples);
	      double random_objective_value = hyperplane_to_solution(h+1, random_solution, &env);
	      printf("Reach = %0.3f\n", random_objective_value);
	      printf("Hyperplane: ");
	      for(int i = 0; i <= samples->dimension+1; i++)
		printf("%g%s", h[i], (i == samples->dimension+1) ? "\n" : " ");

	      printf("checking agreement\n");
	      for(int i = 1; i < samples->dimension+samples_total(samples); i++) {
		int check = h[i] == random_solution[i];
		printf("%3d | % 7.5f | % 7.5f | %d", i, h[i], random_solution[i], check);
		if(!check) {
		  sample_locator_t *loc = locator(i, samples);
		  printf(". s (%s) = (", loc->class ? "+" : "-");
		  for(int j = 0; j < samples->dimension; j++) {
		    printf("%g%s", samples->samples[loc->class][loc->index][j], j == samples->dimension - 1 ? ")\n" : ", ");
		  }
		} else
		  printf("\n");
		  }*/
	      //free(random_solution);
	      //h = random_solution;

	    } else if (param_setting == 2) {
	      //use glpk
	      h = single_run(seed, 120000, &env);
	      printf("Objective = %0.3f\n", h[0]);
	      printf("Hyperplane: ");
	      for(int i = 0; i <= samples->dimension; i++)
		printf("%g%s", h[i], (i == samples->dimension) ? "\n" : " ");
	    } else if (param_setting == 3) {
	      //best random hyperplane
	      env.params->epsilon_positive = 0;
	      env.params->epsilon_negative = 0;
	      env.params->epsilon_precision = 0;

	      srand48(*seed);
	      h = best_random_hyperplane(1, &env);
	      double *random_solution = blank_solution(samples);
	      double random_objective_value = hyperplane_to_solution(h, random_solution, &env);
	      printf("Initial reach = %0.3f\n", random_objective_value);
	      printf("Hyperplane: ");
	      for(int i = 0; i <= dimension; i++)
		printf("%0.5f%s", h[i], (i == dimension) ? "\n" : " ");
	      //free(random_solution);
	      h = random_solution;
	    } else if (param_setting == 4) {
	      //best random unbiased hyperplane
	      srand48(*seed);
	      env.params->epsilon_positive = 0;
	      env.params->epsilon_negative = 0;
	      env.params->epsilon_precision = 0;
	      if(t == 0)
		add_bias(samples);
	      h = best_random_hyperplane_unbiased(1, &env);
	      double *random_solution = blank_solution(samples);
	      double random_objective_value = hyperplane_to_solution(h, random_solution, &env);
	      printf("Initial reach = %0.3f\n", random_objective_value);
	      printf("Hyperplane: ");
	      for(int i = 0; i <= dimension; i++)
		printf("%0.5f%s", h[i], (i == dimension) ? "\n" : " ");
	      //free(random_solution);
	      h = random_solution;
	    } else if (param_setting == 5) {
	      	    env.params->epsilon_positive /= 16;
	    env.params->epsilon_negative /= 16;
	    env.params->epsilon_precision /= 16;
	    add_bias(samples);
	    //normalize_samples(samples);

	    /*gurobi_param p2 = {
	      .threads = 1,
	      .MIPFocus = 0,
	      .ImproveStartGap = 0,
	      .ImproveStartTime = GRB_INFINITY,
	      .VarBranch = -1,
	      .Heuristics = 0.05,
	      .Cuts = -1,
	      .RINS = -1,
	      .method = 7,
	      .init = NULL
	    };
	    h = single_gurobi_run(seed, 1200000000, 1200, &env, &p2);
	    exit(0);
	    */

	    //testing the full search procedure:
	    int **cones = CALLOC(16, int *);
	    int max_cones = 16;
	    int n_cones = 0;
	    double best_obj = -1;
	    while(1) {
	      if(n_cones + 1 == max_cones) {
		max_cones *= 2;
		cones = realloc(cones, max_cones*sizeof(int *));
	      }
	      if(n_cones == 0) {
		h = best_random_hyperplane_unbiased(1, &env);
		/*gurobi_param p = {
		  .threads = 0,
		  .MIPFocus = 0,
		  .ImproveStartGap = 0,
		  .ImproveStartTime = GRB_INFINITY,
		  .VarBranch = -1,
		  .Heuristics = 0.05,
		  .Cuts = -1,
		  .RINS = -1,
		  .method = 3,
		  .init = NULL
		};
		h = single_gurobi_run(seed, 120000, 1200, &env, &p) + 1;*/
	      } else {
		printf("about to find outside cones\n");
		env.params->epsilon_positive *= 2;
		env.params->epsilon_negative *= 2;
		env.params->epsilon_precision *= 2;
		h = find_outside_cones(seed, 120000, &env, cones, n_cones);
		env.params->epsilon_positive /= 2;
		env.params->epsilon_negative /= 2;
		env.params->epsilon_precision /= 2;
		if(!h) {
		  printf("no soln found outside cones\n");
		  break;
		}
	      }
	      int *cone = expand_cone(&env, h);
	      printf("cone: ");
	      for(int i = 0; i < n; i++) {
		printf("%d ", cone[i]);
	      }
	      printf("\n");
	      //break;

	      gurobi_param p = {
		.threads = 0,
		.MIPFocus = 0,
		.ImproveStartGap = 0,
		.ImproveStartTime = GRB_INFINITY,
		.VarBranch = -1,
		.Heuristics = 0.05,
		.Cuts = -1,
		.RINS = -1,
		.method = 6,
		.init = h,
		.cone = cone
	      };
	      env.params->epsilon_positive *= 2;
	      env.params->epsilon_negative *= 2;
	      env.params->epsilon_precision *= 2;
	      h = single_gurobi_run(seed, 120000, 1200, &env, &p);
	      env.params->epsilon_positive /= 2;
	      env.params->epsilon_negative /= 2;
	      env.params->epsilon_precision /= 2;
	      if(!h) {
		printf("no solution found in cone\n");
		break;
	      }
	      best_obj = fmax(h[0], best_obj);

	      cones[n_cones] = cone;
	      n_cones++;

	      printf("----------------------------------- STATUS ---------------------------------------\n");
	      printf("%d cones found so far\n", n_cones);
	      printf("best objective = %g\n", best_obj);
	      printf("----------------------------------------------------------------------------------\n");
	      //if(n_cones == 2) break;
	    }
	    exit(0);
	
	    } else {
	      printf("invalid arg\n"); exit(1);
	    }
	    //remove these in actual testing
	    env.params->epsilon_positive = 1e-10;
	    env.params->epsilon_negative = 1e-10;
	    env.params->epsilon_precision = 1e-10;

	    double *soln = blank_solution(samples);
	    double obj = hyperplane_to_solution(h, soln, &env);
	    printf("Objective = %0.3f\n", obj);

	    printf("Hyperplane: ");
	    for(int i = 1; i <= env.samples->dimension + 1; i++)
	      printf("%0.5f%s", soln[i], (i == env.samples->dimension + 1) ? "\n" : " ");
	    //printf("Objective value: %0.3f\n", h[0]);

	    //env.params = params_default();

	    reaches[k] = reach(soln, env.samples);
	    precisions[k] = precision(soln, env.samples);
	    printf("P = %0.3f\n", precisions[k]);
	    if(isnan(precisions[k])) precisions[k] = 0;
	   
	    printf("Training: %u\t%lg\n", 
		   reaches[k],
		   precisions[k]);

	    double *valid = blank_solution(samples_validation);
	    hyperplane_to_solution_parts(h, valid, env.params, samples_validation);

	    printf("Validation: %u\t%lg\n", reach(valid, samples_validation), precision(valid, samples_validation));

	    exit(0);
	    k++;
            free(h);
	    free(soln);
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
    printf("Average reach/cluster size = %0.3f, average precision = %0.3f\n", avg_reach/(((double) n)/10), avg_prec);
    free(reaches);
    free(precisions);

    exp_res_t res;
    res.prec = avg_prec;
    res.reach = avg_reach;
    return res;
}

int main(int argc, char *argv[]) {
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
