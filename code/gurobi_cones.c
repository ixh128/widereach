#include "widereach.h"
#include "helper.h"

#define TRY_MODEL(premise, step) \
  TRY(premise, \
      error_handle(state, model, step), \
      NULL)
#define TRY_STATE(body) TRY(int state = body, state != 0, state)
#define TRY_MODEL_ON(premise) TRY(*state = premise, *state != 0, NULL)

#define NAME_LEN_MAX 255
#define MSG_LEN 256

GRBmodel *init_gurobi_cones_model(int *state, const env_t *env) {
  GRBenv *p = NULL;
  
  TRY_MODEL_ON(GRBemptyenv(&p));

  TRY_MODEL_ON(GRBsetintparam(p, "NonConvex", 2));
  
  TRY_MODEL_ON(GRBstartenv(p));

  params_t *params = env->params;
  GRBmodel *model = NULL;
  TRY_MODEL_ON(
    GRBnewmodel(p, &model, params->name, 0, NULL, NULL, NULL, NULL, NULL));
  
  TRY_MODEL_ON(
    GRBsetintattr(model, GRB_INT_ATTR_MODELSENSE, GRB_MAXIMIZE));

  return model; 
}

/*int add_gurobi_cones_hyperplane(GRBmodel *model, const env_t *env) {
  TRY_STATE(add_gurobi_hyperplane(model, env->samples->dimension));
  
}

int add_gurobi_cones_sample_constr(GRBmodel *model, sample_locator_t locator, int label, char *name, const env_t *env);

int add_gurobi_cones_sample(GRBmodel *model, sample_locator_t locator, const env_t *env) {
  int label = env->samples->label[locator.class];
  char name[NAME_LEN_MAX];
  snprintf(name, NAME_LEN_MAX, "%c%u", 
          label_to_varname(label), 
          (unsigned int) locator.index + 1);
  
  // Add sample decision variable
  TRY_STATE(add_gurobi_sample_var(model, label, name));
      
  // Add sample constraint
  return add_gurobi_cones_sample_constr(model, locator, label, name, env);

}

int gurobi_cones_accumulator(samples_t *samples, sample_locator_t locator, void *model, void *env) {
  return add_gurobi_cones_sample((GRBmodel *) model, locator, (const env_t *) env);
}

int add_gurobi_cones_samples(GRBmodel *model, const env_t *env) {
  return reduce(env->samples, (void *) model, gurobi_cones_accumulator, (void *) env);
  }*/

int gurobi_cones_accumulator(samples_t *samples, sample_locator_t locator, void *model, void *env) {
  int label = ((env_t *) env)->samples->label[locator.class];
  char name[NAME_LEN_MAX];
  snprintf(name, NAME_LEN_MAX, "%c%u", label_to_varname(label), (unsigned int) locator.index + 1);
  return add_gurobi_sample_var(model, label, name);
}

int add_gurobi_cones_vars(GRBmodel *model, const env_t *env) {
  int d = env->samples->dimension;
  int n = samples_total(env->samples);
  //add hyperplane vars (indices [0, d-1])
  double *lb = CALLOC(d, double);
  double *ub = CALLOC(d, double);
  char **varnames = CALLOC(d, char *);
  char *name;
  for(int i = 0; i < d; i++) {
    lb[i] = -GRB_INFINITY;
    ub[i] = GRB_INFINITY;
    name = varnames[i] = CALLOC(NAME_LEN_MAX, char);
    snprintf(name, NAME_LEN_MAX, "v%u", i + 1);
  }
  TRY_STATE(GRBaddvars(model, d, 0, NULL, NULL, NULL, NULL, lb, ub, NULL, varnames));
  for(int i = 0; i < d; i++)
    free(varnames[i]);
  free(lb); free(ub); free(varnames);

  //add sample vars (indices [d, d+n-1])
  TRY_STATE(reduce(env->samples, (void *) model, gurobi_cones_accumulator, (void *)env));

  //add matrix vars (indices [d+n, d+n+d*d-1], cij at index d+n+i*d+j, for 0-indexed i and j)
  lb = CALLOC(d*d, double);
  ub = CALLOC(d*d, double);
  varnames = CALLOC(d*d, char *);
  for(int i = 0; i < d; i++) {
    for(int j = 0; j < d; j++) {
      int k = i*d+j;
      lb[k] = -GRB_INFINITY;
      ub[k] = GRB_INFINITY;
      name = varnames[k] = CALLOC(NAME_LEN_MAX, char);
      snprintf(name, NAME_LEN_MAX, "c%u_%u", i+1, j+1);
    }
  }
  TRY_STATE(GRBaddvars(model, d*d, 0,  NULL, NULL, NULL, NULL, lb, ub, NULL, varnames));
  for(int k = 0; k < d*d; k++)
    free(varnames[k]);
  free(lb); free(ub); free(varnames);

  //add etas (indices [d+n+d*d, d+2*n+d*d-1], eta_i at index d+n+d*d+i, for 0-indexed i)
  lb = CALLOC(n, double);
  ub = CALLOC(n, double);
  varnames = CALLOC(n, char *);
  char *vtypes = CALLOC(n, char);
  for(int i = 0; i < n; i++) {
    lb[i] = 0;
    ub[i] = 1;
    name = varnames[i] = CALLOC(NAME_LEN_MAX, char);
    snprintf(name, NAME_LEN_MAX, "e%u", i);
    vtypes[i] = GRB_BINARY;
  }
  TRY_STATE(GRBaddvars(model, n, 0, NULL, NULL, NULL, NULL, lb, ub, vtypes, varnames));
  free(lb); free(ub); free(varnames); free(vtypes);

  //add gamma (index d+2*n+d*d)
  TRY_STATE(GRBaddvar(model, 0, NULL, NULL, 0, 0, 1, GRB_BINARY, "g"));

  //add V (index d+2*n+d*d+1)
  //TRY_STATE(GRBaddvar(model, 0, NULL, NULL, -env->params->lambda, env->params->violation_type ? 0 : -GRB_INFINITY, GRB_INFINITY, GRB_CONTINUOUS, "V"));
  //force V to be 0 (i.e. disable lagrangian relaxation)
  TRY_STATE(GRBaddvar(model, 0, NULL, NULL, -env->params->lambda, 0, 0, GRB_CONTINUOUS, "V"));

  return 0;
}

gsl_matrix **get_proj_terms(const env_t *env) {
  //returns an array of matrices uiuiT/uiTui, for each sample ui
  int d = env->samples->dimension;
  int n = samples_total(env->samples);
  gsl_matrix **terms = CALLOC(n, gsl_matrix *);
  vsamples_t *vs = samples_to_vec(env->samples);
  int k = 0;
  for(size_t class = 0; class < vs->class_cnt; class++) {
    for(size_t i = 0; i < vs->count[class]; i++) {
      gsl_vector *u = vs->samples[class][i];
      double denom;
      gsl_blas_ddot(u, u, &denom);
      terms[k] = gsl_matrix_calloc(u->size, u->size);
      gsl_blas_dger(1/denom, u, u, terms[k]);
      /*printf("k = %d:\n", k);
      printf("u = (");
      for(int j = 0; j < d; j++)
	printf("%g%s", gsl_vector_get(u, j), j == d - 1 ? ")\n" : ", ");
      printf("c = \n");
      print_matrix(stdout, terms[k]);*/
      k++;
    }
  }
  return terms;
}

int add_gurobi_cones_constrs(GRBmodel *model, const env_t *env) {
  int d = env->samples->dimension;
  int n = samples_total(env->samples);

  int nvars;
  GRBgetintattr(model, GRB_INT_ATTR_NUMVARS, &nvars);
  printf("nvars = %d\n", nvars);
  printf(" idx  | name \n");
  printf("-------------\n");
  for(int i = 0; i < nvars; i++) {
    char *name;
    GRBgetstrattrelement(model, GRB_STR_ATTR_VARNAME, i, &name);
    printf(" %4d | %s\n", i, name);
  }
  //add hyperplane constraint
  int *lind = CALLOC(d+1, int);
  double *lval = CALLOC(d+1, double);
  int *qrow = CALLOC(d, int);
  int *qcol = CALLOC(d, int);
  double *qval = CALLOC(d, double);
  for(int j = 0; j < d; j++) {
    for(int i = 0; i < d; i++) {
      lind[i] = d+n+j*d+i; //index of cji
      lval[i] = 1;

      qrow[i] = d+2*n+d*d; //index of gamma
      qcol[i] = d+n+j*d+i; //index of cji
      qval[i] = -2;

      //if using gamma in {-1, 0, 1}:
      /*qrow[i] = d+2*n+d*d;
      qcol[i] = d+n+j*d+i;
      qval[i] = -1;*/
    }
    lind[d] = j; //index of vj
    lval[d] = 10;
    char name[NAME_LEN_MAX];
    snprintf(name, NAME_LEN_MAX, "Cv%d", j+1);
    TRY_STATE(GRBaddqconstr(model, d+1, lind, lval, d, qrow, qcol, qval, GRB_EQUAL, 0, name));
  }
  free(lind); free(lval); free(qrow); free(qcol); free(qval);

  //add cij constraints
  gsl_matrix **projterms = get_proj_terms(env);
  lind = CALLOC(n+1, int);
  lval = CALLOC(n+1, double);
  for(int i = 0; i < d; i++) {
    for(int j = 0; j < d; j++) {
      double rhs = i == j ? 1 : 0;
      for(int k = 0; k < n; k++) {
	lind[k] = d+n+d*d+k; //eta_k
	lval[k] = gsl_matrix_get(projterms[k], i, j);
      }
      lind[n] = d+n+i*d+j; //cij
      lval[n] = 1;
      char name[NAME_LEN_MAX];
      snprintf(name, NAME_LEN_MAX, "Cc%u_%u", i+1, j+1);
      TRY_STATE(GRBaddconstr(model, n+1, lind, lval, GRB_EQUAL, rhs, name));
    }
  }
  free(lind); free(lval);
  for(int k = 0; k < n; k++)
    gsl_matrix_free(projterms[k]);
  free(projterms);

  //add eta constraint
  lind = CALLOC(n, int);
  lval = CALLOC(n, double);
  for(int k = 0; k < n; k++) {
    lind[k] = d+n+d*d+k; //eta_k
    lval[k] = 1;
    char *varname;
    GRBgetstrattrelement(model, GRB_STR_ATTR_VARNAME, d+n+d*d+k, &varname);
    //printf("k = %d => %s\n", k, varname);
  }
  TRY_STATE(GRBaddconstr(model, n, lind, lval, GRB_EQUAL, d-1, "eta"));
  free(lind); free(lval);

  //add sample constraints
  lind = CALLOC(1, int);
  lval = CALLOC(1, double);
  qrow = CALLOC(d*d, int);
  qcol = CALLOC(d*d, int);
  qval = CALLOC(d*d, double);
  for(size_t class = 0; class < env->samples->class_cnt; class++) {
    for(size_t index = 0; index < env->samples->count[class]; index++) {
      int label = env->samples->label[class];
      lind[0] = idx(0, class, index, env->samples)-2; //minus 2 because of 0-indexing and the lack of bias
      lval[0] = label;
      int k = 0;
      for(int i = 0; i < d; i++) {
	for(int j = 0; j < d; j++) {
	  qrow[k] = d+n+j*d+i; //cji
	  qcol[k] = i; //vi
	  qval[k] = env->samples->samples[class][index][j]*(-label);
	  k++;
	}
      }
      char name[NAME_LEN_MAX];
      snprintf(name, NAME_LEN_MAX, "%c%u", label_to_varname(label), (unsigned int) index + 1);
      TRY_STATE(GRBaddqconstr(model, 1, lind, lval, d*d, qrow, qcol, qval, GRB_LESS_EQUAL, label_to_bound(label, env->params), name));
    }
  }
  free(lind); free(lval); free(qrow); free(qcol); free(qval);

  //add precision constraint
  lind = CALLOC(n+1, int);
  lval = CALLOC(n+1, double);
  sparse_vector_t *constraint = sparse_vector_blank(samples_total(env->samples)+2);
  double theta = env->params->theta;
  for(size_t class = 0; class < env->samples->class_cnt; class++) {
    int label = env->samples->label[class];
    double penalty = label > 0 ? theta - 1 : theta;
    cover_row(constraint, class, penalty, env->samples);
  }
  for(int i = 1; i <= constraint->len; i++)
    constraint->ind[i] -= 2;
  append(constraint, d+2*n+d*d+1, -1);

  return GRBaddconstr(model, constraint->len, constraint->ind+1, constraint->val+1, GRB_LESS_EQUAL, -theta * env->params->epsilon_precision, "V");

}

GRBmodel *gurobi_cones_milp(int *state, const env_t *env) {
  GRBmodel *model = init_gurobi_cones_model(state, env);

  TRY_MODEL_ON(NULL == model);
  printf("initialized\n");
  TRY_MODEL_ON(add_gurobi_cones_vars(model, env));
  GRBupdatemodel(model);
  printf("added vars\n");
  TRY_MODEL_ON(add_gurobi_cones_constrs(model, env));
  printf("added constrs\n");

  return model;
}

double *single_gurobi_cones_run(unsigned int *seed, 
                          int tm_lim, 
                          int tm_lim_tune, 
                          env_t *env) {
    samples_t *samples = env->samples;
    env->solution_data = solution_data_init(samples_total(samples));
    
    if (seed != NULL) {
      printf("Applying seed %u\n", *seed);
      //srand48(*seed);
    }

    int state;
    GRBmodel *model;

    printf("about to create model\n");
    TRY_MODEL(model = gurobi_cones_milp(&state, env), "model creation");
    printf("created model\n");

    TRY_MODEL(state = GRBsetstrparam(GRBgetenv(model), "LogFile", "gurobi_log.log"), "set log file");

    TRY_MODEL(state = GRBupdatemodel(model), "update model");

    int nvars;
    TRY_MODEL(state = GRBgetintattr(model, GRB_INT_ATTR_NUMVARS, &nvars), "get number of variables");
    
    TRY_MODEL(
      state = GRBsetdblparam(GRBgetenv(model), 
                             "TuneTimeLimit", 
                             tm_lim_tune / 1000.),
      "set time limit for tuning")
    printf("optimize ...\n");
    
    TRY_MODEL(
      state = GRBsetdblparam(GRBgetenv(model), 
                             "TimeLimit", 
                             tm_lim / 1000.),
      "set time limit")

      /*GRBwrite(model, "pre.prm"); //save params before tuning

    TRY_MODEL(state = GRBtunemodel(model), "autotune");
    int nresults;
    TRY_MODEL(state = GRBgetintattr(model, "TuneResultCount", &nresults), "get tune results");
    if(nresults > 0)
      TRY_MODEL(state = GRBgettuneresult(model, 0), "apply tuning");
      GRBwrite(model, "post.prm");*/
    
    int optimstatus;
    int dimension = env->samples->dimension;
        
    GRBwrite(model, "model.lp");

    TRY_MODEL(state = GRBoptimize(model), "optimize");
    TRY_MODEL(state = GRBcomputeIIS(model), "compute IIS");
    GRBwrite(model, "tmp.ilp");
    
    //find optimal value of decision vars
    printf("%d variables\n", nvars);
    
    //NOTE: modified - should be nvars and not include the objval
    double *result = CALLOC(nvars+1, double);
    TRY_MODEL(state = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, result), "get objective value");
    TRY_MODEL(state = GRBgetdblattrarray(model, GRB_DBL_ATTR_X, 0, nvars, result+1), "get decision vars");

    /*double *result = CALLOC(nvars, double);
      TRY_MODEL(state = GRBgetdblattrarray(model, GRB_DBL_ATTR_X, 0, nvars, result), "get decision vars");*/

    /*printf("Decision vars:\n");
    for(int i = 0; i < nvars; i++)
    printf("%0.3f%s", result[i], (i == nvars - 1) ? "\n" : " ");*/

    GRBwrite(model, "soln.sol");

    GRBfreeenv(GRBgetenv(model));
    GRBfreemodel(model);

    return result;
}
