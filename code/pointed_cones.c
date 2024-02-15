#include "widereach.h"
#include "helper.h"

#include <gsl/gsl_rng.h>
#include <gsl/gsl_linalg.h>

//this file implements a linear program to detect whether a cone is pointed
//it is used in the greer tree algorithm

#define E(premise) {error = premise; if(error) ERR();}

int error;
GRBenv *pc_env;

void ERR() {
  printf("ERROR CODE: %d\nMESSAGE: %s\n", error, GRBgeterrormsg(pc_env));
  exit(1);
}

GRBmodel *create_pc_model() {
  GRBenv *env = NULL;

  E(GRBemptyenv(&env));

  E(GRBsetintparam(env, "OutputFlag", 0));
  
  E(GRBstartenv(env));

  GRBmodel *model = NULL;
  E(GRBnewmodel(env, &model, "pointed cone", 0, NULL, NULL, NULL, NULL, NULL));

  E(GRBsetintattr(model, GRB_INT_ATTR_MODELSENSE, GRB_MAXIMIZE));

  pc_env = env;

  return model;
}

void add_pc_vars(GRBmodel *model, vsamples_t *vs) {
  //add the d+1 decision variables to the LP

  //first, add the decision variables corresponding to x:
  size_t d = vs->dimension;

  double *lb = CALLOC(d, double);
  double *ub = CALLOC(d, double);
  char **varnames = CALLOC(d, char *);
  for(int i = 0; i < d; i++) {
    lb[i] = -GRB_INFINITY;
    ub[i] = GRB_INFINITY;
    varnames[i] = CALLOC(10, char);
    snprintf(varnames[i], 10, "x%d", i);
  }

  E(GRBaddvars(model, d, 0, NULL, NULL, NULL, NULL, lb, ub, NULL, varnames));

  //then gamma:
  E(GRBaddvar(model, 0, NULL, NULL, 1, 0, 1, GRB_CONTINUOUS, "g"));

  free(lb);
  free(ub);
  for(int i = 0; i < d; i++) free(varnames[i]);
  free(varnames);
}

void add_pc_sample_constr(GRBmodel *model, gsl_vector *s, char *name) {
  //add the constraint corresponding to a single sample s

  int d = s->size;
  double *x = CALLOC(d, double);
  for(int i = 0; i < d; i++)
    x[i] = gsl_vector_get(s, i);

  sparse_vector_t *v = to_sparse(d, x, 1);

  //append -gamma
  append(v, d+1, -1);

  //convert to gurobi indices (equivalent to gurobi_indices, could refactor)
  int vlen = v->len;
  for (int i = 1; i <= vlen; i++) {
    v->ind[i]--;
  }

  E(GRBaddconstr(model,
		 v->len, v->ind+1, v->val+1, 
		 GRB_GREATER_EQUAL, 0,
		 name));

  free(x);
  free(delete_sparse_vector(v));
}

void add_pc_constrs(GRBmodel *model, vsamples_t *vs, hplane_data vd) {
  //adds the constraints to the LP
  //adds one constraint for each sample y such that <x, y> = 0, where x = vd.v

  for(int i = 0; i < vd.res.npz; i++) {
    add_pc_sample_constr(model, vs->samples[1][vd.res.pz[i]], NULL);
  }
  for(int i = 0; i < vd.res.nnz; i++) {
    gsl_vector *flipped = gsl_vector_alloc(vd.v->size);
    gsl_vector_memcpy(flipped, vs->samples[0][vd.res.nz[i]]);
    gsl_vector_scale(flipped, -1);
    add_pc_sample_constr(model, flipped, NULL);
    gsl_vector_free(flipped);
  }
}

pc_soln detect_pointed_cone(vsamples_t *vs, hplane_data vd) {
  //solves the LP described in Greer 2.3.33

  int d = vs->dimension;

  GRBmodel *model = create_pc_model();
  add_pc_vars(model, vs);
  add_pc_constrs(model, vs, vd);

  E(GRBwrite(model, "tmp_pc.lp")) //TODO: remove

  E(GRBoptimize(model));

  double gamma, *x_arr = CALLOC(d, double);

  E(GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, &gamma));
  E(GRBgetdblattrarray(model, GRB_DBL_ATTR_X, 0, d, x_arr));

  E(GRBfreemodel(model));
  GRBfreeenv(pc_env);

  gsl_vector *x = gsl_vector_alloc(d);
  for(int i = 0; i < d; i++)
    gsl_vector_set(x, i, x_arr[i]);

  free(x_arr);
  return (pc_soln) {.gamma = gamma, .x = x};
}
