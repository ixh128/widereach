#include "widereach.h"
#include "helper.h"
#include <math.h>
#include <string.h>

#include <gsl/gsl_cdf.h>

#define NAME_LEN_MAX 255

#define TRY_MODEL(condition) TRY(, condition, NULL)
#define TRY_STATE(body) TRY(int state = body, state != 0, state)
#define TRY_MODEL_ON(premise) TRY(*state = premise, *state != 0, NULL)

GRBmodel *init_gurobi_model(int *state, const env_t *env) {
  GRBenv *p = NULL;
  
  TRY_MODEL_ON(GRBemptyenv(&p));

  //TRY_MODEL_ON(GRBsetdblparam(p, "MemLimit", 8));

  //TRY_MODEL_ON(GRBsetintparam(p, "OutputFlag", 0));
  
  TRY_MODEL_ON(GRBstartenv(p));

  params_t *params = env->params;
  GRBmodel *model = NULL;
  TRY_MODEL_ON(
    GRBnewmodel(p, &model, params->name, 0, NULL, NULL, NULL, NULL, NULL));
  
  TRY_MODEL_ON(
    GRBsetintattr(model, GRB_INT_ATTR_MODELSENSE, GRB_MAXIMIZE));

  return model; 
}

int add_gurobi_hyperplane(GRBmodel *model, size_t dimension, int biased) {
  int dimension_int = (int) dimension;
  int hyperplane_cnt = 1 + dimension_int;
  double *lb = CALLOC(hyperplane_cnt, double);
  char **varnames = CALLOC(hyperplane_cnt, char *);
  char *name;
  double *ub = CALLOC(hyperplane_cnt, double);
  for (int i = 0; i < hyperplane_cnt; i++) {
    lb[i] = -GRB_INFINITY;
    ub[i] = GRB_INFINITY;
    name = varnames[i] = CALLOC(NAME_LEN_MAX, char);
    snprintf(name, NAME_LEN_MAX, "w%u", i + 1);
  }
  snprintf(varnames[dimension_int], NAME_LEN_MAX, "c");

  if(biased == 0) {
    lb[hyperplane_cnt-1]=0;
    ub[hyperplane_cnt-1]=0;
  }
  
  return GRBaddvars(model, hyperplane_cnt, 
                          0, NULL, NULL, NULL, 
                          NULL, lb, ub, 
                          NULL, 
                          varnames);
}


int add_gurobi_sample_var(GRBmodel *model, int label, char *name) {
  /*TRY_STATE(
    GRBaddvar(model, 0, NULL, NULL, 
              label_to_obj(label), 
              0., 1., GRB_BINARY, 
              name));

  //return GRBupdatemodel(model); //TODO: figure out why this was here
  return 0;*/
  return GRBaddvar(model, 0, NULL, NULL, 
              label_to_obj(label), 
              0., 1., GRB_BINARY, 
		   name);
}

int add_gurobi_sample_var_cont(GRBmodel *model, int label, char *name) {
  return GRBaddvar(model, 0, NULL, NULL, 
              label_to_obj(label), 
              0., 1., GRB_CONTINUOUS, 
		   name);
}

int add_gurobi_thr_var(GRBmodel *model, char *name) {
  char tname[50] = "t";
  strcat(tname, name);

  return GRBaddvar(model, 0, NULL, NULL, 0, 0, 1, GRB_BINARY, tname);
}

int add_gurobi_thr_constr(GRBmodel *model, char *name) {
  char tname[50] = "t";
  strcat(tname, name);

  int tind, xind;
  TRY_STATE(GRBgetvarbyname(model, name, &xind));
  TRY_STATE(GRBgetvarbyname(model, tname, &tind));

  double xpts[] = {0, 0.499, 0.501, 1};
  double ypts[] = {0, 0, 1, 1};
  
  return GRBaddgenconstrPWL(model, tname, xind, tind, 4, xpts, ypts);
}

// Convert index set from GLPK to Gurobi format
void gurobi_indices(sparse_vector_t *v) {
  int vlen = v->len;
  for (int i = 1; i <= vlen; i++) {
    v->ind[i]--;
  }
}

int add_gurobi_sample_constr(
    GRBmodel *model, 
    sample_locator_t locator,
    int label, 
    char *name,
    const env_t *env) {
  // Set coefficients of w
  samples_t *samples = env->samples;
  int dimension = (int) samples->dimension;
  size_t class = locator.class;
  size_t sample_index = locator.index;
  sparse_vector_t *v = 
    to_sparse(dimension, samples->samples[class][sample_index], 2);

  // Set coefficient of c
  append(v, dimension + 1, -1.); 
  // Change sign depending on sample class
  multiply(v, -label);
  // Add sample decision variable
  int col_idx = idx(0, class, sample_index, samples);
  append(v, col_idx, label);
  
  gurobi_indices(v);
  
  return GRBaddconstr(model,
                      v->len, v->ind+1, v->val+1, 
                      GRB_LESS_EQUAL, label_to_bound(label, env->params),
                      name);
}

int add_gurobi_sample_constr_bilinear(
    GRBmodel *model, 
    sample_locator_t locator,
    int label, 
    char *name,
    const env_t *env) {
  // Set coefficients of w
  samples_t *samples = env->samples;
  int dimension = (int) samples->dimension;
  size_t class = locator.class;
  size_t sample_index = locator.index;
  sparse_vector_t *v = 
    to_sparse(dimension, samples->samples[class][sample_index], 2);
  int col_idx = idx(0, class, sample_index, samples);   //sample decision variable

  

  int *qcol = CALLOC(dimension+1, int);
  for(int i = 0; i < dimension+1; i++) {
    qcol[i] = col_idx - 1; //-1 due to gurobi indices
  }

  //printf("adding col idx %d\n", col_idx);
  
  // Set coefficient of c
  append(v, dimension + 1, -1.); 
  // Change sign depending on sample class
  //multiply(v, -label);

  //append(v, col_idx, label);
  
  gurobi_indices(v);

  //epsilon term is linear:

  int lnz;
  int *lind;
  double *lval;
  double rhs;
  if(label > 0) {
    rhs = 0;
    lnz = 1;
    lind = CALLOC(1, int);
    lind[0] = col_idx-1;
    lval = CALLOC(1, double);
    lval[0] = label > 0 ? -env->params->epsilon_positive : env->params->epsilon_negative;
  } else {
    rhs = -env->params->epsilon_negative;
    lnz = dimension+2;
    lind = CALLOC(lnz, int);
    lval = CALLOC(lnz, double);
    int i;
    for(i = 0; i < dimension+1; i++) {
      lind[i] = v->ind[i+1];
      lval[i] = -v->val[i+1];
    }
    lind[i] = col_idx-1;
    lval[i] = env->params->epsilon_negative;
  }

  /*printf("now lind[0] = %d\n", lind[0]);

  printf("var %s\n", name);
  printf("v->len = %d\n", v->len);

  for(int i = 0; i < dimension+1; i++) {
    printf("i = %d => v->ind[i+1] = %d, v->val[i+1] = %g, qcol[i] = %d\n", i, v->ind[i+1], v->val[i+1], qcol[i]);
  }
  for(int j = 0; j < lnz; j++) {
    printf("j = %d => lind[j] = %d, lval[j] = %g\n", j, lind[j], lval[j]);

    }*/
  
  /*return GRBaddconstr(model,
                      v->len, v->ind+1, v->val+1, 
                      GRB_LESS_EQUAL, label_to_bound(label, env->params),
                      name);*/
  return GRBaddqconstr(model, lnz, lind, lval, v->len, v->ind+1, qcol, v->val+1, GRB_GREATER_EQUAL, 0, name);
}


int add_gurobi_sample_constr_qp(
    GRBmodel *model, 
    sample_locator_t locator,
    int label, 
    char *name,
    const env_t *env) {
  // Set coefficients of w
  samples_t *samples = env->samples;
  int dimension = (int) samples->dimension;
  size_t class = locator.class;
  size_t sample_index = locator.index;
  sparse_vector_t *v = 
    to_sparse(dimension, samples->samples[class][sample_index], 2);

  double c = 1;
  multiply(v, c);

  // Set coefficient of c
  //append(v, dimension + 1, -1.); 
  // Change sign depending on sample class
  multiply(v, -label);
  // Add sample decision variable
  int col_idx = idx(0, class, sample_index, samples);
  append(v, col_idx, label);
  
  gurobi_indices(v);
  
  return GRBaddconstr(model,
                      v->len, v->ind+1, v->val+1, 
                      GRB_LESS_EQUAL, label_to_bound(label, env->params),
                      name);
}


int add_gurobi_sample_constr_qp1(GRBmodel *model, 
				sample_locator_t locator,
				int label, 
				char *name,
				const env_t *env) {
  // Set coefficients of w
  samples_t *samples = env->samples;
  int dimension = (int) samples->dimension;
  size_t class = locator.class;
  size_t sample_index = locator.index;

  double *hplane_coefs = CALLOC(dimension, double); //to become -0.5s
  for(size_t i = 0; i < dimension; i++) {
    double c = 100;
    hplane_coefs[i] = -0.5*c*samples->samples[class][sample_index][i];
    //hplane_coefs[i] = -0.5*samples->samples[class][sample_index][i];
  }
  
  sparse_vector_t *v = 
    to_sparse(dimension, hplane_coefs, 2);

  // Set coefficient of c (this can probably be removed, since we force c = 0)
  //append(v, dimension + 1, -1.); 
  // Change sign depending on sample class
  multiply(v, label);
  // Add sample decision variable
  int col_idx = idx(0, class, sample_index, samples);
  append(v, col_idx, label);
  
  gurobi_indices(v);
  
  return GRBaddconstr(model,
                      v->len, v->ind+1, v->val+1, 
                      GRB_LESS_EQUAL, label/2.,
                      name);
}

int add_gurobi_sample_constr_sep(GRBmodel *model, 
				 sample_locator_t locator,
				 int label, 
				 char *name,
				 const env_t *env) {
  // Set coefficients of w
  samples_t *samples = env->samples;
  int dimension = (int) samples->dimension;
  size_t class = locator.class;
  size_t sample_index = locator.index;
  sparse_vector_t *v = 
    to_sparse(dimension, samples->samples[class][sample_index], 1);

  // Set coefficient of c
  // append(v, dimension + 1, -1.); 
  // Change sign depending on sample class
  multiply(v, label);
  // Add sample decision variable
  int col_idx = idx(0, class, sample_index, samples);
  append(v, col_idx, 1);
  
  gurobi_indices(v);
  
  return GRBaddconstr(model,
                      v->len, v->ind+1, v->val+1, 
                      GRB_GREATER_EQUAL, 1,
                      name);
}

int add_gurobi_sample(GRBmodel *model, 
                      sample_locator_t locator, 
                      const env_t *env) {
  int label = env->samples->label[locator.class];
  char name[NAME_LEN_MAX];
  snprintf(name, NAME_LEN_MAX, "%c%u", 
          label_to_varname(label), 
          (unsigned int) locator.index + 1);
  
  // Add sample decision variable
  TRY_STATE(add_gurobi_sample_var(model, label, name));
        
  // Add sample constraint
  return add_gurobi_sample_constr(model, locator, label, name, env);
}

int add_gurobi_sample_bilinear(GRBmodel *model, 
                      sample_locator_t locator, 
                      const env_t *env) {
  int label = env->samples->label[locator.class];
  char name[NAME_LEN_MAX];
  snprintf(name, NAME_LEN_MAX, "%c%u", 
          label_to_varname(label), 
          (unsigned int) locator.index + 1);
  
  // Add sample decision variable
  TRY_STATE(add_gurobi_sample_var(model, label, name));
        
  // Add sample constraint
  return add_gurobi_sample_constr_bilinear(model, locator, label, name, env);
}


int add_gurobi_sample_qp(GRBmodel *model, 
			 sample_locator_t locator, 
			 const env_t *env) {
  int label = env->samples->label[locator.class];
  char name[NAME_LEN_MAX];
  snprintf(name, NAME_LEN_MAX, "%c%u", 
          label_to_varname(label), 
          (unsigned int) locator.index + 1);
  
  // Add sample decision variable
  TRY_STATE(add_gurobi_sample_var_cont(model, label, name));
      
  // Add sample constraint
  return add_gurobi_sample_constr_qp(model, locator, label, name, env);
}

int add_gurobi_sample_sep(GRBmodel *model, 
			  sample_locator_t locator, 
			  const env_t *env) {
  int label = env->samples->label[locator.class];
  char name[NAME_LEN_MAX];
  snprintf(name, NAME_LEN_MAX, "t%c%u", 
          label_to_varname(label), 
          (unsigned int) locator.index + 1);
  
  // Add sample decision variable
  TRY_STATE(GRBaddvar(model, 0, NULL, NULL, 
              1, 
              0, GRB_INFINITY, GRB_CONTINUOUS, 
		   name););
      
  // Add sample constraint
  return add_gurobi_sample_constr_sep(model, locator, label, name, env);
}


int add_gurobi_sample_thr(GRBmodel *model, 
                      sample_locator_t locator, 
                      const env_t *env) {
  int label = env->samples->label[locator.class];
  char name[NAME_LEN_MAX];
  snprintf(name, NAME_LEN_MAX, "%c%u", 
          label_to_varname(label), 
          (unsigned int) locator.index + 1);
  
  // Add sample decision variable
  //TRY_STATE(add_gurobi_sample_var(model, label, name));

  
    //TODO: Threshold vars seem to cause an issue, since the index of the variable V has been shifted by them being added
    //could just move them into the add-precision function for a hacky soln
  TRY_STATE(add_gurobi_thr_var(model, name));

  TRY_STATE(GRBupdatemodel(model));
  
  // Add sample constraint
  return add_gurobi_thr_constr(model, name);
}

int gurobi_accumulator(
    samples_t *samples, 
    sample_locator_t locator, 
    void *model, 
    void *env) {
  return add_gurobi_sample((GRBmodel *) model, locator, (const env_t *) env);
}

int gurobi_accumulator_bilinear(
    samples_t *samples, 
    sample_locator_t locator, 
    void *model, 
    void *env) {
  return add_gurobi_sample_bilinear((GRBmodel *) model, locator, (const env_t *) env);
}

int gurobi_accumulator_qp(
    samples_t *samples, 
    sample_locator_t locator, 
    void *model, 
    void *env) {
  return add_gurobi_sample_qp((GRBmodel *) model, locator, (const env_t *) env);
}

int gurobi_accumulator_sep(
    samples_t *samples, 
    sample_locator_t locator, 
    void *model, 
    void *env) {
  return add_gurobi_sample_sep((GRBmodel *) model, locator, (const env_t *) env);
}

int gurobi_accumulator_thr(
    samples_t *samples, 
    sample_locator_t locator, 
    void *model, 
    void *env) {
  return add_gurobi_sample_thr((GRBmodel *) model, locator, (const env_t *) env);
}

int add_gurobi_samples(GRBmodel *model, const env_t *env) {
  return reduce(env->samples, (void *) model, gurobi_accumulator, (void *) env); 
}

int add_gurobi_samples_qp(GRBmodel *model, const env_t *env) {
  return reduce(env->samples, (void *) model, gurobi_accumulator_qp, (void *) env); 
}

int add_gurobi_samples_sep(GRBmodel *model, const env_t *env) {
  return reduce(env->samples, (void *) model, gurobi_accumulator_sep, (void *) env); 
}

int add_gurobi_samples_thr(GRBmodel *model, const env_t *env) {
  return reduce(env->samples, (void *) model, gurobi_accumulator_thr, (void *) env); 
}

int add_gurobi_samples_bilinear(GRBmodel *model, const env_t *env) {
  return reduce(env->samples, (void *) model, gurobi_accumulator_bilinear, (void *) env); 
}


int add_gurobi_precision(GRBmodel *model, const env_t *env) {
  params_t *params = env->params;
  TRY_STATE(
    GRBaddvar(model, 0, NULL, NULL, 
              -params->lambda, 
              params->violation_type ? 0. : -GRB_INFINITY, 
              GRB_INFINITY,
              GRB_CONTINUOUS, "V"));
  
  double theta = params->theta;
  sparse_vector_t *constraint = precision_row(env->samples, theta);
  gurobi_indices(constraint);
  return GRBaddconstr(model, 
                      constraint->len, constraint->ind+1, constraint->val+1, 
                      GRB_LESS_EQUAL, -theta * params->epsilon_precision, 
                      "V");
}

int add_gurobi_precision_strict(GRBmodel *model, const env_t *env) {
  params_t *params = env->params;
  /*TRY_STATE(
    GRBaddvar(model, 0, NULL, NULL, 
              -params->lambda, 
              params->violation_type ? 0. : -GRB_INFINITY, 
              GRB_INFINITY,
              GRB_CONTINUOUS, "V"));*/
  TRY_STATE(
    GRBaddvar(model, 0, NULL, NULL, 
	      0, 
              params->violation_type ? 0. : -GRB_INFINITY, 
              GRB_INFINITY,
              GRB_CONTINUOUS, "V"));
  
  double theta = params->theta;
  sparse_vector_t *constraint = precision_row_strict(env->samples, theta);
  gurobi_indices(constraint);
  return GRBaddconstr(model, 
                      constraint->len, constraint->ind+1, constraint->val+1, 
                      GRB_LESS_EQUAL, -theta * params->epsilon_precision, 
                      "V");
}

int add_gurobi_precision_thr(GRBmodel *model, const env_t *env) {
  params_t *params = env->params;
  TRY_STATE(
    GRBaddvar(model, 0, NULL, NULL, 
              -params->lambda, 
              params->violation_type ? 0. : -GRB_INFINITY, 
              GRB_INFINITY,
              GRB_CONTINUOUS, "V"));

  TRY_STATE(GRBupdatemodel(model));

  TRY_STATE(add_gurobi_samples_thr(model, env));
  
  double theta = params->theta;
  sparse_vector_t *constraint = precision_row_thr(env->samples, theta);
  gurobi_indices(constraint);
  printf("About to add constraint\n");
  printf("Constraint: ");
  for(int i = 0; i < constraint->len; i++) {
    printf("[ind = %d, val = %0.3f] ", constraint->ind[i], constraint->val[i]);
  }
  printf("STOP\n");
  int nvars;
  TRY_STATE(GRBgetintattr(model, GRB_INT_ATTR_NUMVARS, &nvars));
  printf("%d vars\n", nvars);
  GRBwrite(model, "tmp1.lp");
  return GRBaddconstr(model, 
                      constraint->len, constraint->ind+1, constraint->val+1, 
                      GRB_LESS_EQUAL, -theta * params->epsilon_precision, 
                      "V");
}

GRBmodel *gurobi_milp(int *state, const env_t *env) {
    samples_t *samples = env->samples;
	TRY_MODEL(!is_binary(samples))
	GRBmodel *model = init_gurobi_model(state, env);
    TRY_MODEL(NULL == model)
	
      TRY_MODEL_ON(add_gurobi_hyperplane(model, samples->dimension, 1))

    TRY_MODEL_ON(add_gurobi_samples(model, env))
        
      TRY_MODEL_ON(add_gurobi_precision(model, env))
      int nvars;

    GRBupdatemodel(model);
    /*GRBgetintattr(model, GRB_INT_ATTR_NUMVARS, &nvars);
  printf("nvars = %d\n", nvars);
  printf(" idx  | name \n");
  printf("-------------\n");
  for(int i = 0; i < nvars; i++) {
    char *name;
    GRBgetstrattrelement(model, GRB_STR_ATTR_VARNAME, i, &name);
    printf(" %4d | %s\n", i, name);
    }*/
      /*      *state = add_gurobi_precision_thr(model, env);
    printf("State = %d\n", *state);
    printf("Added precision");*/
    
 	// p = add_valid_constraints(p, env);

    GRBwrite(model, "tmp.lp");
	return model;
}

int add_gurobi_bound_unit_hplane(GRBmodel *model, const env_t *env, int norm);

GRBmodel *gurobi_milp_unbiased(int *state, const env_t *env) {
  samples_t *samples = env->samples;
  TRY_MODEL(!is_binary(samples));
  GRBmodel *model = init_gurobi_model(state, env);
  TRY_MODEL(NULL == model);
  
  TRY_MODEL_ON(add_gurobi_hyperplane(model, samples->dimension, 0));

  TRY_MODEL_ON(add_gurobi_samples(model, env));
        
  TRY_MODEL_ON(add_gurobi_precision(model, env));
  int nvars;

  //TRY_MODEL_ON(add_gurobi_bound_unit_hplane(model, env, 1));

  GRBupdatemodel(model);
  GRBwrite(model, "tmp.lp");
  return model;
}


int add_gurobi_aux_qp_vars(GRBmodel *model, const env_t *env) {
  //add violation var
  params_t *params = env->params;
  TRY_STATE(
    GRBaddvar(model, 0, NULL, NULL, 
              -params->lambda, 
              params->violation_type ? 0. : -GRB_INFINITY, 
              GRB_INFINITY,
              GRB_CONTINUOUS, "V"));

  //adds quadratic vars for the precision constraint
  TRY_STATE(GRBaddvar(model, 0, NULL, NULL, 0, -GRB_INFINITY, 0, GRB_CONTINUOUS, "L1"));
  TRY_STATE(GRBaddvar(model, 0, NULL, NULL, 0, -GRB_INFINITY, GRB_INFINITY, GRB_CONTINUOUS, "Q1"));
  TRY_STATE(GRBaddvar(model, 0, NULL, NULL, 0, -GRB_INFINITY, GRB_INFINITY, GRB_CONTINUOUS, "Q2"));

  int iL1 = violation_idx(0, env->samples)+1; //index of L1, etc
  int iQ1 = iL1 + 1;
  int iQ2 = iL1 + 2;
  printf("[%d, %d], [%d]\n", iQ1, iQ2, iL1);

  double theta = env->params->theta;
  sparse_vector_t *cL1 = sparse_vector_blank(samples_total(env->samples)+1);
  for(size_t class = 0; class < env->samples->class_cnt; class++) {
    double coef;
    if(env->samples->label[class] > 0)
      coef = -1;
    else
      coef = theta/(1-theta);
    cover_row(cL1, class, coef, env->samples);
  }
  append(cL1, iL1, -1);
  gurobi_indices(cL1);
  TRY_STATE(GRBaddconstr(model, cL1->len, cL1->ind+1, cL1->val+1, GRB_EQUAL, 0, "cL1"));

  //is it a memory leak to leave the sparse vector unfreed? the old functions do it

  sparse_vector_t *q1q = sparse_vector_blank(positives(env->samples));
  cover_row(q1q, 1, -1, env->samples); //all xs^2 get coef = -1
  sparse_vector_t *q1l = sparse_vector_blank(positives(env->samples)+1);
  cover_row(q1l, 1, 1, env->samples); //all xs get coef 1
  append(q1l, iQ1, -1);
  gurobi_indices(q1q);
  gurobi_indices(q1l);
  TRY_STATE(GRBaddqconstr(model, q1l->len, q1l->ind+1, q1l->val+1, q1q->len, q1q->ind+1, q1q->ind+1, q1q->val+1, GRB_EQUAL, 0, "cQ1"));

  sparse_vector_t *q2q = sparse_vector_blank(negatives(env->samples));
  cover_row(q2q, 0, -1, env->samples); //all ys^2 get coef = -1
  sparse_vector_t *q2l = sparse_vector_blank(negatives(env->samples)+1);
  cover_row(q2l, 0, 1, env->samples); //all ys get coef 1
  append(q2l, iQ2, -1);
  gurobi_indices(q2q);
  gurobi_indices(q2l);
  return GRBaddqconstr(model, q2l->len, q2l->ind+1, q2l->val+1, q2q->len, q2q->ind+1, q2q->ind+1, q2q->val+1, GRB_EQUAL, 0, "cQ2");
}

int add_gurobi_precision_qp(GRBmodel *model, const env_t *env) {
  int iL1 = violation_idx(0, env->samples) + 1; //index of L1, etc
  int iQ1 = iL1 + 1;
  int iQ2 = iL1 + 2;

  int relaxation_coeff = 1; //set to 0 to disable relaxation

  double epsi = gsl_cdf_ugaussian_Qinv(env->params->epsilon_prob);
  printf("epsi = %g\n", epsi);
  double theta = env->params->theta;
  sparse_vector_t *l = sparse_vector_blank(3);
  append(l, iQ1, epsi*epsi);
  append(l, iQ2, epsi*epsi*(theta/(1-theta))*(theta/(1-theta)));
  append(l, violation_idx(0, env->samples), -relaxation_coeff);
  printf("appended viol idx\n");
  
  sparse_vector_t *q = sparse_vector_blank(1);
  append(q, iL1, -1);

  gurobi_indices(l);
  gurobi_indices(q);
  int state = GRBaddqconstr(model, l->len, l->ind+1, l->val+1, q->len, q->ind+1, q->ind+1, q->val+1, GRB_LESS_EQUAL, 0, "qP");
  
  return state;
}

int add_gurobi_unit_hplane(GRBmodel *model, const env_t *env) {
  //constraint ||w|| = 1 (2-norm)
  /*size_t d = env->samples->dimension;
  sparse_vector_t *q = sparse_vector_blank(d);
  for(int i = 0; i < d; i++) {
    append(q, i, 1);
  }

  return GRBaddqconstr(model, 0, NULL, NULL, q->len, q->ind+1, q->ind+1, q->val+1, GRB_EQUAL, 1, "w");*/
  GRBsetintparam(GRBgetenv(model), "NonConvex", 2);
  size_t d = env->samples->dimension;
  /*sparse_vector_t *q = sparse_vector_blank(d);
  for(int i = 0; i < d; i++) {
    append(q, i, 1);
  }
  return GRBaddqconstr(model, 0, NULL, NULL, q->len, q->ind+1, q->ind+1, q->val+1, GRB_LESS_EQUAL, 1, "w");*/

  TRY_STATE(GRBupdatemodel(model));

  TRY_STATE(GRBaddvar(model, 0, NULL, NULL, 0, 1, GRB_INFINITY, GRB_CONTINUOUS, "r"));

  int idx;
  TRY_STATE(GRBgetintattr(model, "NumVars", &idx));
  //getting the index of the last-added var, which is r
  //we would subtract 1 from it, except that we update the model before making r, so it doesn't count toward the index yet

  int *ind = CALLOC(d, int);
  for(int i = 0; i < d; i++)
    ind[i] = i;

  return GRBaddgenconstrNorm(model, "norm", idx, d, ind, 2);
}

int add_gurobi_bound_unit_hplane(GRBmodel *model, const env_t *env, int norm) {
  //constraint ||w|| <= 1
  //norm = 0, 1, 2, or GRB_INFINITY
  size_t d = env->samples->dimension;
  /*sparse_vector_t *q = sparse_vector_blank(d);
  for(int i = 0; i < d; i++) {
    append(q, i, 1);
  }
  return GRBaddqconstr(model, 0, NULL, NULL, q->len, q->ind+1, q->ind+1, q->val+1, GRB_LESS_EQUAL, 1, "w");*/

  TRY_STATE(GRBupdatemodel(model));

  TRY_STATE(GRBaddvar(model, 0, NULL, NULL, 0, 0, 1, GRB_CONTINUOUS, "r"));

  int idx;
  TRY_STATE(GRBgetintattr(model, "NumVars", &idx));
  //getting the index of the last-added var, which is r
  //we would subtract 1 from it, except that we update the model before making r, so it doesn't count toward the index yet

  int *ind = CALLOC(d, int);
  for(int i = 0; i < d; i++)
    ind[i] = i;

  return GRBaddgenconstrNorm(model, "norm", idx, d, ind, norm);
}

GRBmodel *gurobi_qp(int *state, const env_t *env) {
    samples_t *samples = env->samples;
    TRY_MODEL(!is_binary(samples))
      GRBmodel *model = init_gurobi_model(state, env);
    TRY_MODEL(NULL == model)
	
      TRY_MODEL_ON(add_gurobi_hyperplane(model, samples->dimension, 0))

      env->params->cont = 1;

    TRY_MODEL_ON(add_gurobi_samples_qp(model, env))

    GRBsetintparam(GRBgetenv(model), "NonConvex", 2);

    TRY_MODEL_ON(add_gurobi_aux_qp_vars(model, env));

        
    TRY_MODEL_ON(add_gurobi_precision_qp(model, env))

      TRY_MODEL_ON(add_gurobi_unit_hplane(model, env));

      GRBwrite(model, "tmpqp.lp");

    GRBupdatemodel(model);
	return model;
}

int gurobi_cones_miqp_accumulator(samples_t *samples, sample_locator_t locator, void *modela, void *enva) {
  GRBmodel *model = modela;
  env_t *env = enva;
  char name[NAME_LEN_MAX];
  snprintf(name, NAME_LEN_MAX, "v%c%u",
	   label_to_varname(env->samples->label[locator.class]),
          (unsigned int) locator.index + 1);

  return GRBaddvar(model, 0, NULL, NULL, 0, 0, 1, GRB_BINARY, name);
}

int add_gurobi_cones(GRBmodel *model, const env_t *env) {
  samples_t *samples = env->samples;
  
  //add binary vars vs:
  TRY_STATE(reduce(samples, (void *) model, gurobi_cones_miqp_accumulator, (void *) env));

  int start_v = violation_idx(0, samples) + 1; //index where the vs start
  size_t n = samples_total(samples);
  size_t d = samples->dimension;
  
  //set branching priority of vs
  int *ones = CALLOC(n, int);
  for(int i = 0; i < n; i++)
    ones[i] = 10;
  TRY_STATE(GRBsetintattrarray(model, "BranchPriority", start_v-1, n, ones));
  free(ones);

  //add constraint on the vs:
  sparse_vector_t *cv = sparse_vector_blank(samples_total(samples));
  for(size_t i = start_v; i < start_v + n; i++) {
    append(cv, i, 1);
  }
  gurobi_indices(cv);
  TRY_STATE(GRBaddconstr(model, cv->len, cv->ind+1, cv->val+1, GRB_GREATER_EQUAL, n - (d-1), "cv")); //could also be EQUAL to only explore leaf nodes
  TRY_STATE(GRBaddconstr(model, cv->len, cv->ind+1, cv->val+1, GRB_LESS_EQUAL, n - 1, "cv")); //this way, it must be in the tree

  free(delete_sparse_vector(cv));

  //add linear constraints -v <= w . s <= v
  for(int i = 0; i < n - 1; i++) {
    sample_locator_t *loc = locator(i + d + 2, env->samples);
    int vi = i + start_v-1;
    sparse_vector_t *l1 = sparse_vector_blank(d + 1);
    sparse_vector_t *l2 = sparse_vector_blank(d + 1);
    for(int j = 0; j < d; j++) {
      append(l1, j, samples->samples[loc->class][loc->index][j]);
      append(l2, j, samples->samples[loc->class][loc->index][j]);
    }
    append(l1, vi, -1);
    append(l2, vi, 1);
    /*gurobi_indices(l1);
      gurobi_indices(l2);*/

    int label = env->samples->label[loc->class];
    char name1[NAME_LEN_MAX];
    char name2[NAME_LEN_MAX];
    snprintf(name1, NAME_LEN_MAX, "u%c%u", 
	     label_to_varname(label), 
	     (unsigned int) loc->index + 1);
    snprintf(name2, NAME_LEN_MAX, "l%c%u", 
	     label_to_varname(label), 
	     (unsigned int) loc->index + 1);

    //Tight bounds:
    TRY_STATE(GRBaddconstr(model, l1->len, l1->ind+1, l1->val+1, GRB_LESS_EQUAL, 0, name1));
    TRY_STATE(GRBaddconstr(model, l2->len, l2->ind+1, l2->val+1, GRB_GREATER_EQUAL, 0, name2));

    /*//with leeway:
    double eps = 0.03;

    TRY_STATE(GRBaddconstr(model, l1->len, l1->ind+1, l1->val+1, GRB_LESS_EQUAL, eps, name1));
    TRY_STATE(GRBaddconstr(model, l2->len, l2->ind+1, l2->val+1, GRB_GREATER_EQUAL, -eps, name2));*/


    free(delete_sparse_vector(l1));
    free(delete_sparse_vector(l2));
  }

  /*//add quadratic constraints on v w . s
  for(int i = 0; i < n-1; i++) {
    sample_locator_t *loc = locator(i + d + 2, env->samples);
    int vi = i + start_v;
    sparse_vector_t *qrow = sparse_vector_blank(d);
    sparse_vector_t *qcol = sparse_vector_blank(d); //values will be stored in here
    for(int j = 0; j < d; j++) {
      append(qrow, vi, 1);
      append(qcol, j, samples->samples[loc->class][loc->index][j]);
    }
    gurobi_indices(qrow);

    int label = env->samples->label[loc->class];
    char name[NAME_LEN_MAX];
    snprintf(name, NAME_LEN_MAX, "o%c%u", 
	     label_to_varname(label), 
	     (unsigned int) loc->index + 1);
    
    TRY_STATE(GRBaddqconstr(model, 0, NULL, NULL, qrow->len, qrow->ind+1, qcol->ind+1, qcol->val+1, GRB_EQUAL, 0, name));

    free(delete_sparse_vector(qrow));
    free(delete_sparse_vector(qcol));
    }*/
  //disable presolve, otherwise gurobi removes all of that
  //only necessary with the >= bound
  //TRY_STATE(GRBsetintparam(GRBgetenv(model), "Presolve", 0));
  return 0;
}

int add_gurobi_unit_hplane_relaxed(GRBmodel *model, const env_t *env) {
  size_t d = env->samples->dimension;
  double mult = positives(env->samples) + 1;
  int idxs[d];
  double vals[d];
  for(int i = 0; i < d; i++) {
    idxs[i] = i;
    vals[i] = -mult;
  }
  return GRBaddqpterms(model, d, idxs, idxs, vals);
}

GRBmodel *gurobi_cones_miqp(int *state, const env_t *env) { 
    samples_t *samples = env->samples;
    TRY_MODEL(!is_binary(samples))
      GRBmodel *model = init_gurobi_model(state, env);
    TRY_MODEL(NULL == model)
	
      TRY_MODEL_ON(add_gurobi_hyperplane(model, samples->dimension, 0))

    TRY_MODEL_ON(add_gurobi_samples(model, env))

      //GRBsetintparam(GRBgetenv(model), "NonConvex", 2);

    TRY_MODEL_ON(add_gurobi_precision(model, env))

      TRY_MODEL_ON(add_gurobi_cones(model, env));

    //TRY_MODEL_ON(add_gurobi_unit_hplane(model, env));
    //TRY_MODEL_ON(add_gurobi_unit_hplane_relaxed(model, env));
    TRY_MODEL_ON(add_gurobi_bound_unit_hplane(model, env, 1));

    GRBwrite(model, "cones_qp.lp");

    GRBupdatemodel(model);
	return model; 
}

GRBmodel *gurobi_relaxation(int *state, const env_t *env) {
    samples_t *samples = env->samples;
    TRY_MODEL(!is_binary(samples));
    GRBmodel *model = init_gurobi_model(state, env);
    TRY_MODEL(NULL == model);
    
    TRY_MODEL_ON(add_gurobi_hyperplane(model, samples->dimension, 0));
    
    TRY_MODEL_ON(add_gurobi_samples(model, env));
        
    TRY_MODEL_ON(add_gurobi_precision_strict(model, env));

    GRBsetintparam(GRBgetenv(model), "NonConvex", 2);
    TRY_MODEL_ON(add_gurobi_unit_hplane(model, env));

    GRBupdatemodel(model);

    //convert integer vars to double
    int nvars;
    TRY_MODEL_ON(GRBgetintattr(model, GRB_INT_ATTR_NUMVARS, &nvars));
    char *conts = CALLOC(nvars, char);
    char *vtypes = CALLOC(nvars, char);
    TRY_MODEL_ON(GRBgetcharattrarray(model, "VType", 0, nvars, vtypes));
    for(int i = 0; i < nvars; i++)
      conts[i] = GRB_CONTINUOUS;
    TRY_MODEL_ON(GRBsetcharattrarray(model, "VType", 0, nvars, conts));

    free(conts);
    free(vtypes);

    GRBwrite(model, "tmp.lp");
    return model;
}

int gurobi_constrain_side(GRBmodel *model, const env_t *env, sample_locator_t loc, int side) {
  //constrains sample to be on the side of the hyperplane given by side (= +/-1)
  samples_t *samples = env->samples;
  int dimension = (int) samples->dimension;
  size_t class = loc.class;
  size_t sample_index = loc.index;
  sparse_vector_t *v = 
    to_sparse(dimension, samples->samples[class][sample_index], 0);

  gurobi_indices(v);

  char name[NAME_LEN_MAX];
  snprintf(name, NAME_LEN_MAX, "%c%c%u", side == 1 ? 'p' : 'n', label_to_varname(env->samples->label[loc.class]), (unsigned int) loc.index + 1);

  if(side == 1) {
    return GRBaddconstr(model,
			v->len, v->ind+1, v->val+1, 
			GRB_GREATER_EQUAL, env->params->epsilon_positive, name);
  } else {
    return GRBaddconstr(model,
			v->len, v->ind+1, v->val+1, 
			GRB_LESS_EQUAL, -env->params->epsilon_negative, name);
  }
}

int add_gurobi_small_cone_constr(GRBmodel *model, const env_t *env, double *w, sample_locator_t loc) {
  samples_t *samples = env->samples;
  double *s = samples->samples[loc.class][loc.index];
  double dot = 0;
  for(int j = 0; j < samples->dimension; j++)
    dot += w[j]*s[j];
  //printf("sample (%d, %lu): %g\n", loc.class, loc.index, dot);
  if(loc.class == 1 && dot >= env->params->epsilon_positive) {
    //constrain w . s >= eps
    //printf("  adding + constr\n");
    return gurobi_constrain_side(model, env, loc, 1);
  }
  if(loc.class == 0 && dot <= env->params->epsilon_negative) {
    //constrain w . s <= -eps
    //printf("  adding - constr\n");
    return gurobi_constrain_side(model, env, loc, -1);
  }
  return 0;
}

int add_gurobi_small_cone_constrs(GRBmodel *model, const env_t *env, double *w) {
  samples_t *samples = env->samples;
  int state = 0;
  /*printf("hyperplane: ");
  for(int i = 0; i < samples->dimension; i++)
  printf("%g%s", w[i], i == samples->dimension - 1 ? "\n" : " ");*/
  for(int class = 0; class < 2; class++) {
    for(size_t i = 0; i < samples->count[class]; i++) {
      sample_locator_t loc = {class, i};
      state |= add_gurobi_small_cone_constr(model, env, w, loc);
    }
  }
  return state;
}

GRBmodel *gurobi_relax_within_small_cone(int *state, const env_t *env, double *w) {
    samples_t *samples = env->samples;
    TRY_MODEL(!is_binary(samples));
    GRBmodel *model = init_gurobi_model(state, env);
    TRY_MODEL(NULL == model);
    
    TRY_MODEL_ON(add_gurobi_hyperplane(model, samples->dimension, 0));
    
    TRY_MODEL_ON(add_gurobi_samples(model, env));
        
    //TRY_MODEL_ON(add_gurobi_precision_strict(model, env));

    TRY_MODEL_ON(add_gurobi_small_cone_constrs(model, env, w));

    /*GRBsetintparam(GRBgetenv(model), "NonConvex", 2);
      TRY_MODEL_ON(add_gurobi_unit_hplane(model, env));*/

    GRBupdatemodel(model);

    //convert integer vars to double
    int nvars;
    TRY_MODEL_ON(GRBgetintattr(model, GRB_INT_ATTR_NUMVARS, &nvars));
    char *conts = CALLOC(nvars, char);
    char *vtypes = CALLOC(nvars, char);
    TRY_MODEL_ON(GRBgetcharattrarray(model, "VType", 0, nvars, vtypes));
    for(int i = 0; i < nvars; i++)
      conts[i] = GRB_CONTINUOUS;
    TRY_MODEL_ON(GRBsetcharattrarray(model, "VType", 0, nvars, conts));

    free(conts);
    free(vtypes);

    GRBwrite(model, "tmp.lp");
    return model;
}

int add_gurobi_cone_constrs(GRBmodel *model, const env_t *env, int *cone) {
  samples_t *samples = env->samples;
  int state = 0;
  /*printf("hyperplane: ");
  for(int i = 0; i < samples->dimension; i++)
  printf("%g%s", w[i], i == samples->dimension - 1 ? "\n" : " ");*/
  size_t n = samples_total(env->samples);
  size_t d = env->samples->dimension;
  for(int i = 0; i < n; i++) {
    //printf("i = %d => cone[i] = %d\n", i, cone[i]);
    if(cone[i] == 0) continue;
    sample_locator_t *loc = locator(i+d+2, samples);
    state |= gurobi_constrain_side(model, env, *loc, cone[i]);
  }
  return state;
}

GRBmodel *gurobi_relax_within_cone(int *state, const env_t *env, int *cone) {
    samples_t *samples = env->samples;
    TRY_MODEL(!is_binary(samples));
    GRBmodel *model = init_gurobi_model(state, env);
    TRY_MODEL(NULL == model);
    
    TRY_MODEL_ON(add_gurobi_hyperplane(model, samples->dimension, 0));
    
    TRY_MODEL_ON(add_gurobi_samples(model, env));
        
    //TRY_MODEL_ON(add_gurobi_precision_strict(model, env));

    TRY_MODEL_ON(add_gurobi_cone_constrs(model, env, cone));

    /*GRBsetintparam(GRBgetenv(model), "NonConvex", 2);
      TRY_MODEL_ON(add_gurobi_unit_hplane(model, env));*/

    GRBupdatemodel(model);

    //convert integer vars to double
    /*int nvars;
    TRY_MODEL_ON(GRBgetintattr(model, GRB_INT_ATTR_NUMVARS, &nvars));
    char *conts = CALLOC(nvars, char);
    char *vtypes = CALLOC(nvars, char);
    TRY_MODEL_ON(GRBgetcharattrarray(model, "VType", 0, nvars, vtypes));
    for(int i = 0; i < nvars; i++)
      conts[i] = GRB_CONTINUOUS;
      TRY_MODEL_ON(GRBsetcharattrarray(model, "VType", 0, nvars, conts));

    free(conts);
    free(vtypes);*/

    GRBwrite(model, "tmp_cone.lp");
    return model;
}

int add_gurobi_ortho(GRBmodel *model, const env_t *env, sample_locator_t loc) {
  samples_t *samples = env->samples;
  int dimension = (int) samples->dimension;
  size_t class = loc.class;
  size_t sample_index = loc.index;
  sparse_vector_t *v = 
    to_sparse(dimension, samples->samples[class][sample_index], 0);

  gurobi_indices(v);

  char name[NAME_LEN_MAX];
  snprintf(name, NAME_LEN_MAX, "o%c%u", label_to_varname(env->samples->label[loc.class]), (unsigned int) loc.index + 1);

  return GRBaddconstr(model,
			v->len, v->ind+1, v->val+1, 
			GRB_EQUAL, 0, name);
}

GRBmodel *gurobi_relax_within_subspace(int *state, const env_t *env, sample_locator_t *basis, int ss_dim) {
    samples_t *samples = env->samples;
    TRY_MODEL(!is_binary(samples));
    GRBmodel *model = init_gurobi_model(state, env);
    TRY_MODEL(GRBsetintparam(GRBgetenv(model), "OutputFlag", 0));
    TRY_MODEL(NULL == model);
    
    TRY_MODEL_ON(add_gurobi_hyperplane(model, samples->dimension, 0));
    
    TRY_MODEL_ON(add_gurobi_samples(model, env));
        
    TRY_MODEL_ON(add_gurobi_precision(model, env));

    for(int i = 0; i < ss_dim; i++) {
      TRY_MODEL_ON(add_gurobi_ortho(model, env, basis[i]));
    }

    TRY_MODEL_ON(add_gurobi_bound_unit_hplane(model, env, 1));

    GRBupdatemodel(model);

    //convert integer vars to double
    int nvars;
    TRY_MODEL_ON(GRBgetintattr(model, GRB_INT_ATTR_NUMVARS, &nvars));
    char *conts = CALLOC(nvars, char);
    char *vtypes = CALLOC(nvars, char);
    TRY_MODEL_ON(GRBgetcharattrarray(model, "VType", 0, nvars, vtypes));
    for(int i = 0; i < nvars; i++)
      conts[i] = GRB_CONTINUOUS;
    TRY_MODEL_ON(GRBsetcharattrarray(model, "VType", 0, nvars, conts));

    free(conts);
    free(vtypes);

    //GRBwrite(model, "tmp_ortho.lp");
    return model;
}

int add_gurobi_outside_cone_vars(GRBmodel *model, const env_t *env, int *cone, int n_supp, int cone_idx) {
  char *vtypes = CALLOC(n_supp, char);
  char **names = CALLOC(n_supp, char *);
  for(int i = 0; i < n_supp; i++) {
    vtypes[i] = GRB_BINARY;
    char name[NAME_LEN_MAX];
    snprintf(name, NAME_LEN_MAX, "z%u_%u", cone_idx, i); //has a slightly misleading name - maybe i should be replaced by a sample idx
    names[i] = name;
  }

  int res = GRBaddvars(model, n_supp, 0, NULL, NULL, NULL, NULL, NULL, NULL, vtypes, names);

  free(vtypes);
  free(names);

  return res;
}

int add_gurobi_outside_cone(GRBmodel *model, const env_t *env, int *cone, int cone_idx) {
  samples_t *samples = env->samples;
  int n = samples_total(samples);
  int d = samples->dimension;
  int n_pos = 0;
  int n_supp = 0;
  for(int i = 0; i < n; i++) {
    if(cone[i] == 1) n_pos++;
    if(cone[i] != 0) n_supp++;
  }
  //TRY_STATE(add_gurobi_outside_cone_vars(model, env, cone, n_supp, cone_idx));
  int *ind = CALLOC(n_supp, int);
  double *val = CALLOC(n_supp, double);
  int k = 0;
  for(int i = 0; i < n; i++) {
    if(cone[i] != 0) {
      sample_locator_t *loc = locator(i+d+2, samples);
      ind[k] = idx(0, loc->class, loc->index, samples) - 1;
      val[k++] = cone[i];
      free(loc);
    }
  }
  char name[NAME_LEN_MAX];
  snprintf(name, NAME_LEN_MAX, "cone%d", cone_idx);
  int res = GRBaddconstr(model, n_supp, ind, val, GRB_LESS_EQUAL, n_pos - 1, name);
  free(ind);
  free(val);
  return res;
}

int add_gurobi_outside_cone_lazy(void *cbdata, const env_t *env, int *cone, int cone_idx) {
  samples_t *samples = env->samples;
  int n = samples_total(samples);
  int d = samples->dimension;
  int n_pos = 0;
  int n_supp = 0;
  for(int i = 0; i < n; i++) {
    if(cone[i] == 1) n_pos++;
    if(cone[i] != 0) n_supp++;
  }
  //TRY_STATE(add_gurobi_outside_cone_vars(model, env, cone, n_supp, cone_idx));
  int *ind = CALLOC(n_supp, int);
  double *val = CALLOC(n_supp, double);
  int k = 0;
  for(int i = 0; i < n; i++) {
    if(cone[i] != 0) {
      sample_locator_t *loc = locator(i+d+2, samples);
      ind[k] = idx(0, loc->class, loc->index, samples) - 1;
      val[k++] = cone[i];
      free(loc);
    }
  }
  int res = GRBcblazy(cbdata, n_supp, ind, val, GRB_LESS_EQUAL, n_pos - 1);
  free(ind);
  free(val);
  return res;
}


int add_gurobi_one_reach(GRBmodel *model, const env_t *env) {
  //constrains the model to have reach >= 1, to avoid nontrivial solutions
  //may lead to infeasibility, although this is actually good for the cone exp search

  int idx_min = idx_extreme(0, 1, 0, env->samples); //first index of positive samples
  size_t P = positives(env->samples);
  int *ind = CALLOC(P, int);
  double *val = CALLOC(P, double);
  for(int i = 0; i < P; i++) {
    ind[i] = i + idx_min - 1;
    val[i] = 1;
  }
  int res = GRBaddconstr(model, P, ind, val, GRB_GREATER_EQUAL, 1, "one");
  free(ind);
  free(val);
  return res;
}

int add_gurobi_samples_forced(GRBmodel *model, const env_t *env) {
  //adds samples with constraints s.t. xs and ys will always be consistent with the hyperplane, regardless of objective
  //may be missing epsilons
  TRY_STATE(add_gurobi_samples(model, env));
  samples_t *samples = env->samples;
  size_t P = positives(env->samples);
  size_t N = negatives(env->samples);
  int dimension = (int) samples->dimension;
  //add constr w . s - c <= xs for all positive samples s
  for(int i = 0; i < P; i++) {
    sample_locator_t locator = {.class = 1, .index = i};
    size_t class = locator.class;
    size_t sample_index = locator.index;
    sparse_vector_t *v = 
      to_sparse(dimension, samples->samples[class][sample_index], 2);

    // Set coefficient of c
    append(v, dimension + 1, -1.); 
    // Add sample decision variable
    int col_idx = idx(0, class, sample_index, samples);
    append(v, col_idx, -1);
  
    gurobi_indices(v);
    char name[NAME_LEN_MAX];
    snprintf(name, NAME_LEN_MAX, "fx%d", i);
    TRY_STATE(GRBaddconstr(model, v->len, v->ind+1, v->val+1, GRB_LESS_EQUAL, -env->params->epsilon_positive, name));
    free(delete_sparse_vector(v));
  }
  //add constr w . s - c > ys - 1 for all negative samples s
  for(int i = 0; i < N; i++) {
    sample_locator_t locator = {.class = 0, .index = i};
    size_t class = locator.class;
    size_t sample_index = locator.index;
    sparse_vector_t *v = 
      to_sparse(dimension, samples->samples[class][sample_index], 2);

    // Set coefficient of c
    append(v, dimension + 1, -1.); 
    // Add sample decision variable
    int col_idx = idx(0, class, sample_index, samples);
    append(v, col_idx, -1);
  
    gurobi_indices(v);
    char name[NAME_LEN_MAX];
    snprintf(name, NAME_LEN_MAX, "fy%d", i);
    TRY_STATE(GRBaddconstr(model, v->len, v->ind+1, v->val+1, GRB_GREATER_EQUAL, env->params->epsilon_negative-1, name));
    free(delete_sparse_vector(v));
  }
  return 0;
}

GRBmodel *gurobi_find_outside_cones(int *state, const env_t *env, int **cones, int n_cones) {
    samples_t *samples = env->samples;
    TRY_MODEL(!is_binary(samples));
    GRBmodel *model = init_gurobi_model(state, env);
    TRY_MODEL(NULL == model);
    
    TRY_MODEL_ON(add_gurobi_hyperplane(model, samples->dimension, 0));
    
    TRY_MODEL_ON(add_gurobi_samples_forced(model, env));
    
    TRY_MODEL_ON(add_gurobi_precision_strict(model, env));

    for(int i = 0; i < n_cones; i++) {
      TRY_MODEL_ON(add_gurobi_outside_cone(model, env, cones[i], i));
    }

    TRY_MODEL_ON(add_gurobi_one_reach(model, env));

    TRY_MODEL_ON(add_gurobi_bound_unit_hplane(model, env, 1));

    TRY_MODEL_ON(GRBupdatemodel(model));

    //set all obj coeffs to 0
    /*int d = env->samples->dimension;
    int p = positives(env->samples);
    double *coeffs = CALLOC(p, double);
    for(int i = 0; i < p; i++)
      coeffs[i] = 0;
    TRY_MODEL_ON(GRBsetdblattrarray(model, "Obj", d+1, p, coeffs));
    free(coeffs);*/

    GRBwrite(model, "tmp_out.lp");
    return model;
}

int add_gurobi_svm_term(GRBmodel *model, size_t d) {
  int *qrow = CALLOC(d, int);
  int *qcol = CALLOC(d, int);
  double *qval = CALLOC(d, double);

  for(int i = 0; i < d; i++) {
    qrow[i] = qcol[i] = i;
    qval[i] = -1;
  }
  int out = GRBaddqpterms(model, d, qrow, qcol, qval);
  free(qrow);
  free(qcol);
  free(qval);
  return out;
}

GRBmodel *gurobi_milp_strict(int *state, const env_t *env) {
  //like gurobi_milp, but forces xs and ys to be consistent with w, no lagrangian relaxation, and forces ||w|| <= 1
    samples_t *samples = env->samples;
    TRY_MODEL(!is_binary(samples));
    GRBmodel *model = init_gurobi_model(state, env);
    TRY_MODEL(NULL == model);
    
    TRY_MODEL_ON(add_gurobi_hyperplane(model, samples->dimension, 0));
    //TRY_MODEL_ON(add_gurobi_svm_term(model, samples->dimension));
    
    //TRY_MODEL_ON(add_gurobi_samples_forced(model, env));
    TRY_MODEL_ON(add_gurobi_samples(model, env));
    
    TRY_MODEL_ON(add_gurobi_precision(model, env));
    //TRY_MODEL_ON(add_gurobi_precision(model, env));

    TRY_MODEL_ON(add_gurobi_bound_unit_hplane(model, env, 2));
    //TRY_MODEL_ON(add_gurobi_unit_hplane(model, env));

    TRY_MODEL_ON(GRBupdatemodel(model));

    //set all obj coeffs to 0
    int d = env->samples->dimension;
    int p = positives(env->samples);
    /*double *coeffs = CALLOC(p, double);
    for(int i = 0; i < p; i++)
      coeffs[i] = 0;
    TRY_MODEL_ON(GRBsetdblattrarray(model, "Obj", d+1, p, coeffs));
    free(coeffs);*/

    GRBwrite(model, "tmp.lp");
    return model;
}

GRBmodel *gurobi_bilinear(int *state, const env_t *env) {
    samples_t *samples = env->samples;
    TRY_MODEL(!is_binary(samples));
    GRBmodel *model = init_gurobi_model(state, env);
    TRY_MODEL(NULL == model);
	
    TRY_MODEL_ON(add_gurobi_hyperplane(model, samples->dimension, 0));

    TRY_MODEL_ON(add_gurobi_samples_bilinear(model, env));
        
    TRY_MODEL_ON(add_gurobi_precision(model, env));

    TRY_MODEL_ON(add_gurobi_bound_unit_hplane(model, env, 2));

    GRBupdatemodel(model);

    GRBwrite(model, "tmp.lp");
	return model;
}


double insep_score(env_t *env) {
  int *state = CALLOC(1, int);
  GRBmodel *model = init_gurobi_model(state, env);

  TRY_STATE(
    GRBsetintattr(model, GRB_INT_ATTR_MODELSENSE, GRB_MINIMIZE));


  TRY_STATE(NULL == model);

  TRY_STATE(add_gurobi_hyperplane(model, env->samples->dimension, 0));

  TRY_STATE(add_gurobi_samples_sep(model, env))

    GRBsetintparam(GRBgetenv(model), "NonConvex", 2);
  
  GRBwrite(model, "sep_test.lp");

    GRBupdatemodel(model);

    TRY_STATE(GRBoptimize(model));

    double obj;
    GRBgetdblattr(model, "ObjVal", &obj);

    size_t d = env->samples->dimension;

    double *w = CALLOC(d, double);
    TRY_STATE(GRBgetdblattrarray(model, GRB_DBL_ATTR_X, 0, d, w));
    printf("hyperplane: ");
    for(int i = 0; i < d; i++)
      printf("%g%s", w[i], i == d - 1 ? "\n" : " ");

    GRBwrite(model, "sep_soln.sol");


    return obj/samples_total(env->samples);
}

double euclid_dist(double *x, double *y, size_t d) {
  double result = 0;
  for(int i = 0; i < d; i++) {
    result += (x[i] - y[i])*(x[i]-y[i]);
  }
  return sqrt(result);
}

double insep_score_euclid(env_t *env) {
  double num_sum = 0, denom_sum = 0;
  samples_t *samples = env->samples;
  size_t d = samples->dimension;
  for(size_t class = 0; class < samples->class_cnt; class++) {
    for(size_t i = 0; i < samples->count[class]; i++) {
      double min_inter = 1e101, min_intra = 1e101;
      
      for(size_t j = 0; j < samples->count[class]; j++) {
	if(j == i) continue;
	double dist = euclid_dist(samples->samples[class][i], samples->samples[class][j], d);
	if(dist < min_inter)
	  min_inter = dist;
      }
      
      for(size_t j = 0; j < samples->count[1-class]; j++) {
	double dist = euclid_dist(samples->samples[class][i], samples->samples[1-class][j], d);
	if(dist < min_intra)
	  min_intra = dist;
      }

      num_sum += min_inter;
      denom_sum += min_intra;
    }
  }
  //no need to divide by n to take averages - would cancel out anyway
  return num_sum/denom_sum;
}
