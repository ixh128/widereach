#include "widereach.h"
#include "helper.h"

#include <memory.h>
#include <sys/time.h>
#include <assert.h>

#define E(premise) {c_error = premise; if(c_error) cone_ERR(__LINE__);}

int c_error;
GRBenv *cone_env;

long long opt_time = 0;

void cone_ERR(int line) {
  printf("On line %d,\nERROR CODE: %d\nMESSAGE: %s\n", line, c_error, GRBgeterrormsg(cone_env));
  exit(1);
}

void init_cone_env() {
  E(GRBemptyenv(&cone_env));

  E(GRBsetintparam(cone_env, "OutputFlag", 0));
  
  E(GRBstartenv(cone_env));
}

long long time_millis() {
    struct timeval tv;

    gettimeofday(&tv,NULL);
    return (((long long)tv.tv_sec)*1000)+(tv.tv_usec/1000);
}

void print_arr(int *a, int n) {
  for(int i = 0; i < n; i++)
    printf("%d ", a[i]);
  printf("\n");
}

GRBmodel *make_cone_model(env_t *env, int *cls) {
  GRBmodel *model = NULL;
  E(GRBnewmodel(cone_env, &model, "support", 0, NULL, NULL, NULL, NULL, NULL));

  E(GRBsetintattr(model, GRB_INT_ATTR_MODELSENSE, GRB_MAXIMIZE));

  int n = samples_total(env->samples);
  int d = env->samples->dimension;

  char **varnames = CALLOC(n, char *);
  for(int i = 0; i < n; i++) {
    varnames[i] = CALLOC(255, char);
    snprintf(varnames[i], 255, "l%u", i+1);
  }
  E(GRBaddvars(model, n, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, varnames));

  int *ind = CALLOC(n, int);
  double *val = CALLOC(n, double);
  for(int i = 0; i < d; i++) {
    for(int j = 0; j < n; j++) {
      ind[j] = j;
      sample_locator_t *loc = locator(j+d+2, env->samples);
      val[j] = env->samples->samples[loc->class][loc->index][i];
      if(cls[j] == 0)
	val[j] *= -1;
      free(loc);
    }
    char *name = CALLOC(255, char);
    snprintf(name, 255, "c%u", i+1);

    E(GRBaddconstr(model, n, ind, val, GRB_EQUAL, 0, name));
    free(name);
  }

  free(ind);
  free(val);
  return model;
}

void remove_cols(env_t *env, GRBmodel *model, int *supp, int n_supp, int idx) {
  //adds a constraint which effectively remove columns corresponding to non-supports and to idx
  //assumes idx is a support hplane
  assert(supp[idx]);
  int n = samples_total(env->samples);
  
  int *ind = CALLOC(n-n_supp+1, int);
  double *val = CALLOC(n-n_supp+1, double);
  int k = 0;
  for(int i = 0; i < n; i++) {
    if(!supp[i] || i == idx) {
      ind[k] = i;
      val[k] = 1;
      k++;
    }
  }
  assert(k == n - n_supp+1);
  
  /*printf("idx = %d\n", idx);
  printf("ind = "); print_arr(ind, n-n_supp+1);
  printf("val = ");
  for(int i = 0; i < n-n_supp+1; i++)
    printf("%g ", val[i]);
    printf("\n");*/
  
  E(GRBaddconstr(model, n-n_supp+1, ind, val, GRB_EQUAL, 0, "d"));
  free(ind);
  free(val);
}

void readd_cols(env_t *env, GRBmodel *model) {
  //removes the constraint added by remove_cols
  int d = env->samples->dimension;
  int ind[] = {d};
  E(GRBdelconstrs(model, 1, ind));
}

void set_cone_rhs(env_t *env, GRBmodel *model, int idx, int cls) {
  //sets the RHS coeffs in the gurobi model to be the given sample
  //cls = +/- 1 depending on how the sample was classified
  int d = env->samples->dimension;
  sample_locator_t *loc = locator(idx+d+2, env->samples);
  double *coeffs = CALLOC(d, double);
  for(int i = 0; i < d; i++)
    coeffs[i] = env->samples->samples[loc->class][loc->index][i]*cls;
  E(GRBsetdblattrarray(model, "RHS", 0, d, coeffs));
  free(coeffs);
  free(loc);
}

int hs_redundant(env_t *env, int *supp, int idx, int *cls) {
  int n = samples_total(env->samples);
  int d = env->samples->dimension;
  GRBmodel *model = NULL;
  E(GRBnewmodel(cone_env, &model, "support", 0, NULL, NULL, NULL, NULL, NULL));

  E(GRBsetintattr(model, GRB_INT_ATTR_MODELSENSE, GRB_MAXIMIZE));

  //TODO: counting number of supports is a little slow
  //but probably will refactor all of this anyway when we remake it to not rebuild the gurobi model every time
  int n_supp = 0;
  for(int i = 0; i < n; i++)
    n_supp += supp[i];
  char **varnames = CALLOC(n_supp, char *);
  for(int i = 0; i < n_supp; i++) {
    varnames[i] = CALLOC(255, char);
    snprintf(varnames[i], 255, "l%u", i+1);
  }
  E(GRBaddvars(model, n_supp, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, varnames));
  free(varnames);

  //adding constraints:
  sample_locator_t *s_loc = locator(idx+d+2, env->samples);
  int *ind = CALLOC(n_supp-1, int); //-1 since we don't include idx
  double *val = CALLOC(n_supp-1, double);
  for(int i = 0; i < d; i++) {
    int k = 0;
    //int sign = s_loc->class ? 1 : -1; //not needed
    for(int j = 0; j < n; j++) {
      if(!supp[j] || j == idx) continue;
      ind[k] = k;
      sample_locator_t *loc = locator(j+d+2, env->samples);
      val[k] = env->samples->samples[loc->class][loc->index][i];
      if(cls[j] == 0)
	val[k] *= -1;
      free(loc);
      k++;
    }
    char *name = CALLOC(255, char);
    snprintf(name, 255, "c%u", i+1);
    /*printf("idx = %d\n", idx);
    printf("ind = "); print_arr(ind, n_supp-1);
    printf("val = ");
    for(int i = 0; i < n; i++)
      printf("%g ", val[i]);
      printf("\n");*/
    E(GRBaddconstr(model, n_supp-1, ind, val, GRB_EQUAL, env->samples->samples[s_loc->class][s_loc->index][i], name));
    free(name);
  }
  free(ind);
  free(val);

  //GRBwrite(model, "supp.lp");

  long long start = time_millis();
  E(GRBoptimize(model));
  opt_time += time_millis() - start;

  //GRBwrite(model, "supp.sol");

  int status;

  E(GRBgetintattr(model, "Status", &status));
  GRBfreemodel(model);

  return status != GRB_INFEASIBLE;
}

int hs_redundant_new(env_t *env, GRBmodel *model, int *supp, int idx, int *cls) {
  int n_supp = 0;
  for(int i = 0; i < samples_total(env->samples); i++)
    if(supp[i]) n_supp++;
  remove_cols(env, model, supp, n_supp, idx);
  set_cone_rhs(env, model, idx, cls[idx] == 1 ? 1 : -1);

  int n = samples_total(env->samples);
  /*printf("testing redundancy of sample %d.\n", idx);
  printf("cls = "); print_arr(cls, n);
  printf("sup = "); print_arr(supp, n);*/

  //GRBwrite(model, "supp.lp");

  long long start = time_millis();
  E(GRBoptimize(model));
  opt_time += time_millis() - start;

  //GRBwrite(model, "supp.sol");
  int status;

  E(GRBgetintattr(model, "Status", &status));
  readd_cols(env, model);

  //printf("result: %sredundant\n", status != GRB_INFEASIBLE ? "" : "not ");
  //if(idx == 5) exit(0);

  return status != GRB_INFEASIBLE;
}

int *get_support(env_t *env, GRBmodel *model, int *cls, int *exclude, int *fixed, int *old_supp) {
  //cls[i] = xi or yi
  //exclude[i] = 1 in order to ignore i 
  //fixed[i] = 1 if i is known to be in the frame, else 0
  //old_supp is the previous support, except the one that was removed - everything that used to be a support vector will still be one (probably don't need fixed anymore then). Can be NULL
  //TODO: maybe should not calloc supp; instead pass as argument
  //could refactor to combine fixed and supp into one argument
  //printf("computing support\n");

  size_t n = samples_total(env->samples);
  int *supp = CALLOC(n, int);
  for(int i = 0; i < n; i++) {
    if(!exclude[i]) supp[i] = 1;
  }
  for(int i = 0; i < n; i++) {
    if(fixed[i] || exclude[i]) {// || (old_supp && old_supp[i])) {
      continue;
    }
    //TODO: should not remake a new gurobi model every time
    //if(hs_redundant(env, supp, i, cls)) {
    if(hs_redundant_new(env, model, supp, i, cls)) {
      supp[i] = 0;
      //printf("found redundant hplane----------------------------------------\n");
      //exit(0);
    }
  }

  return supp;
}

int *expand_cone(env_t *env, double *w) {
  size_t n = samples_total(env->samples);
  size_t d = env->samples->dimension;
  double theta = env->params->theta;
  init_cone_env();

  double *soln = blank_solution(env->samples);
  double obj = hyperplane_to_solution(w, soln, env);
  printf("beginning with objective %g\n", obj);
  printf("hyperplane: ");
  for(int i = 0; i < d; i++) {
    printf("%g ", w[i]);
  }
  printf("\n");
  int *cls = CALLOC(n, int);
  for(int i = 0; i < n; i++) {
    cls[i] = (int) soln[i+d+2]; //TODO: maybe wrong
  }
  /*for(int i = 0; i < n; i++) {
    printf("i = %4d | %d\n", i, cls[i]);
    }*/
  GRBmodel *model = make_cone_model(env, cls);
  //GRBmodel *model = NULL;
  //printf("made base model\n");
  int *fixed = CALLOC(n, int);
  int *exclude = CALLOC(n, int);
  int *supp = get_support(env, model, cls, exclude, fixed, NULL); //technically this use of 'fixed' is not correct (it means something different in this function than in get_support), but it doesn't matter

  //printf("cls = "); print_arr(cls, n);
  //printf("sup = "); print_arr(supp, n);
  int min_tp = reach(soln, env->samples);
  int max_fp = false_positives(soln, env->samples);
  printf("has tp = %d, fp = %d => prec = %g\n", min_tp, max_fp, ((double) min_tp) / (min_tp+max_fp));
  if((min_tp == 0 && max_fp == 0) || ((double) min_tp) / (min_tp+max_fp) < theta) {
    printf("received infeasible solution in expand_cone\n");
    return NULL;
  }

  //TODO: may need to make supp a linked list or queue instead
  //instead, this is currently stupidly slow
  int n_supp = 0;
  for(int i = 0; i < n; i++) {
    n_supp += supp[i];
  }

  while(n_supp > 0) {
    //printf("n_supp = %d\n", n_supp);
    int i;
    for(i = 0; i < n && (supp[i] == 0 || fixed[i] == 1); i++); //find next hyperplane to consider (would be better with a LL)
    //for(i = n-1; i >= 0 && (supp[i] == 0 || fixed[i] == 1); i--); //find next hyperplane to consider (would be better with a LL)
    /*printf("i = %d\n", i);
    printf("sup = "); print_arr(supp, n);
    printf("fix = "); print_arr(fixed, n);
    printf("exc = "); print_arr(exclude, n);
    printf("min tp = %d, max fp = %d => min prec = %g\n", min_tp, max_fp, ((double) min_tp)/(min_tp+max_fp));*/
    if(i == n || supp[i] == 0 || fixed[i] == 1)
      break; //this will happen once all support vectors are fixed
    int class = index_to_class(i+d+2, env->samples);
    if(class != cls[i]) {
      //if misclassified, then we can add it without penalty
      ;
      //printf("i misclassified; adding\n");
    } else if (class == 1 && ((double) min_tp-1)/(min_tp+max_fp-1) >= theta) {
      //losing a positive (we consider 0/0 to be infeasible precision - we aren't interested in solutions with 0 reach anyways, so it doesn't hurt)
      min_tp--;
      //printf("i positive, adding\n");
    } else if (class == 0 && ((double) min_tp)/(min_tp+max_fp+1) >= theta) {
      //gaining a negative
      max_fp++;
      //printf("i negative, adding\n");
    } else {
      //precision too small to support adding it
      fixed[i] = 1;
      //printf("precision too small, not adding. class = %d\n", class);
      continue;
    }
    //remove i's hplane from the support
    exclude[i] = 1;
    //and compute new support
    supp[i] = 0;
    //int *new_supp = get_support(env, cls, exclude, supp); //this is what it was before
    int *new_supp = get_support(env, model, cls, exclude, fixed, supp);
    free(supp);
    supp = new_supp;
    n_supp = 0;
    for(int i = 0; i < n; i++) {
      if(supp[i] && !fixed[i])
	n_supp += 1;
    }
  }

  GRBfreemodel(model); //TODO - maybe don't free these; no need to remake them each time we expand a cone, right?
  GRBfreeenv(cone_env);

  //convert to output format
  //supp[i] = 0 if not a supporting hyperplane
  //otherwise +/- 1 depending on which side the cone lies in
  for(int i = 0; i < n; i++) {
    if(supp[i] == 1 && cls[i] == 0)
      supp[i] = -1;
  }
  free(fixed);
  free(exclude);
  free(cls);
  free(soln);

  printf("total time spent optimizing = %lld\n", opt_time);
  return supp;
}

int *expand_cone_fast(env_t *env, double *w) {
  //same as the alg above, but with fewer calls to hs_redundant
  //actually, maybe not. maybe could optimize it some more
  size_t n = samples_total(env->samples);
  size_t d = env->samples->dimension;
  double theta = env->params->theta;
  init_cone_env();

  double *soln = blank_solution(env->samples);
  double obj = hyperplane_to_solution(w, soln, env);
  printf("beginning with objective %g\n", obj);
  int *cls = CALLOC(n, int);
  for(int i = 0; i < n; i++) {
    cls[i] = (int) soln[i+d+1]; //TODO: maybe wrong
  }
  int *fixed = CALLOC(n, int);
  int *exclude = CALLOC(n, int);
  //int *supp = get_support(env, cls, exclude, fixed); //technically this use of 'fixed' is not correct (it means something different in this function than in get_support), but it doesn't matter

  //printf("cls = "); print_arr(cls, n);
  int min_tp = reach(soln, env->samples);
  int max_fp = false_positives(soln, env->samples);
  if(((double) min_tp) / (min_tp+max_fp) < theta) {
    printf("received infeasible solution in expand_cone\n");
    return NULL;
  }

  //TODO: may need to make supp a linked list or queue instead
  //instead, this is currently stupidly slow
  /*int n_supp = 0;
  for(int i = 0; i < n; i++) {
    n_supp += supp[i];
    }*/

  int *p_supp = CALLOC(n, int); //potential suppport

  while(1) {
    //printf("n_supp = %d\n", n_supp);
    for(int i = 0; i < n; i++) p_supp[i] = 1;
    int i;
    for(i = n - 1; i >= 0; i--) {
      if(exclude[i] || hs_redundant(env, p_supp, i, cls)) {
	assert(!fixed[i]);
	p_supp[i] = 0;
      } else if(fixed[i]) {
	continue;
      } else {
	break;
      }
    }
    printf("i = %d\n", i);
    printf("fix = "); print_arr(fixed, n);
    printf("exc = "); print_arr(exclude, n);
    printf("min tp = %d, max fp = %d => min prec = %g\n", min_tp, max_fp, ((double) min_tp)/(min_tp+max_fp));
    if(i == n || i == -1 || fixed[i] == 1) {
      printf("breaking. fixed[%d] = %d\n", i, fixed[i]);
      break;
    }
    int class = index_to_class(i, env->samples);
    if(class != cls[i]) {
      //if misclassified, then we can add it without penalty
      ;
      printf("i misclassified; adding\n");
    } else if (class == 1 && ((double) min_tp-1)/(min_tp+max_fp-1) >= theta) {
      //losing a positive
      min_tp--;
      printf("i positive, adding\n");
    } else if (class == 0 && ((double) min_tp)/(min_tp+max_fp+1) >= theta) {
      //gaining a negative
      max_fp++;
      printf("i negative, adding\n");
    } else {
      //precision too small to support adding it
      fixed[i] = 1;
      printf("precision too small, not adding\n");
      continue;
    }
    //remove i's hplane from the support
    exclude[i] = 1;
    //and compute new support
    /*supp[i] = 0;
    int *new_supp = get_support(env, cls, exclude, supp);
    free(supp);
    supp = new_supp;
    n_supp = 0;
    for(int i = 0; i < n; i++) {
      if(supp[i] && !fixed[i])
	n_supp += 1;
	}*/
  }
  free(p_supp);

  GRBfreeenv(cone_env);

  //convert to output format
  //supp[i] = 0 if not a supporting hyperplane
  //otherwise +/- 1 depending on which side the cone lies in
  //int *supp = get_support(env, cls, exclude, fixed); //may be a way to avoid this call
  int *supp = NULL; //TODO: use the above line with new call signature
  for(int i = 0; i < n; i++) {
    if(supp[i] == 1 && cls[i] == 0)
      supp[i] = -1;
  }
  printf("total time spent optimizing = %lld\n", opt_time);
  return supp;
}
