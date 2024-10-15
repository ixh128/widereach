#include "widereach.h"
#include "helper.h"
#include <math.h>

#define FREQ 1000
#define LIM 500

extern int nrels, nfeas;

int rel_heur(GRBmodel *model, void *cbdata, int where, void *usrdata) {
  if(where != GRB_CB_MIPNODE) return 0;
  //if(rand() > RAND_MAX/2) return 0;
  int nvars;
  int state;
  env_t *env = usrdata;
  state = GRBgetintattr(model, GRB_INT_ATTR_NUMVARS, &nvars);
  double *h = CALLOC(nvars, double); //relaxation solution
  state = GRBcbget(cbdata, where, GRB_CB_MIPNODE_REL, h);
  double *solution = blank_solution(env->samples);
  double obj = hyperplane_to_solution(h, solution, env);

  double bestobj;
  state = GRBcbget(cbdata, where, GRB_CB_MIPNODE_OBJBST, &bestobj);
  if(obj >= bestobj)
    printf("new best\n");
  state = GRBcbsolution(cbdata, solution, NULL);
  int r = reach(solution, env->samples);
  double p = precision(solution, env->samples);
  //printf("[HEUR] Solution with obj = %0.3f. Reach = %d, prec = %0.3f\n", obj, r, p);
  nrels++;
  if(p >= env->params->theta) nfeas++;
  /*printf("       Hyperplane: ");
  for(int i = 0; i < env->samples->dimension+1; i++) printf("%0.3f ", h[i]);
  printf("\n");*/
  free(h);
  free(solution);
  return 0;
}

double last_obj = -INFINITY;
double last_time = 0;
//int switched = 0;
extern double switch_time;

int time_heur(GRBmodel *model, void *cbdata, int where, void *usrdata) {
  if(switch_time != -1) return 0;
  double time;
  int state = GRBcbget(cbdata, where, GRB_CB_RUNTIME, &time);
  if(time >= last_time + 5) {
    last_time = time;
    double obj;
    state |= GRBcbget(cbdata, where, GRB_CB_MIPNODE_OBJBST, &obj);
    if(obj == last_obj) {
      //switch heuristics param
      GRBterminate(model);
      printf("Plateau, setting heuristics to 1\n");
      switch_time = time;   
    } else {
      last_obj = obj;
    }
  }
  return state;
}

double *last_sol;

int node_heur(GRBmodel *model, void *cbdata, int where, void *usrdata) {
  int state = 0;
  //state |= time_heur(model, cbdata, where, usrdata);
  //state |= rel_heur(model, cbdata, where, usrdata);
  return state;
}

int compare_relaxation(GRBmodel *model, void *cbdata, int where, void *usrdata) {
  //compares the most recent relaxation solution with the new integer solution
  //TODO
  return 0;
}

int sol_heur(GRBmodel *model, void *cbdata, int where, void *usrdata) {
  return compare_relaxation(model, cbdata, where, usrdata);
  // check the agreement between the two most recent integer solutions
  /*int nvars;
  int state;
  env_t *env = usrdata;
  state = GRBgetintattr(model, GRB_INT_ATTR_NUMVARS, &nvars);

  if(last_sol == NULL) {
    last_sol = CALLOC(nvars, double);
  } else {
    double *sol = CALLOC(nvars, double);
    state |= GRBcbget(cbdata, where, GRB_CB_MIPSOL_SOL, sol);
    int agreed = 0;
    for(int i = env->samples->dimension+2; i < nvars; i++) {
      if(last_sol[i] == sol[i]) agreed++;
    }
    printf("%d variables agreed out of %ld\n", agreed, nvars - (env->samples->dimension+2));
  }
  state |= GRBcbget(cbdata, where, GRB_CB_MIPSOL_SOL, last_sol);
  return state;*/
  return 0;
}

int gurobi_callback(GRBmodel *model, void *cbdata, int where, void *usrdata) {
  switch(where) {
  case GRB_CB_MIPNODE:
    //exploring a new node
    return node_heur(model, cbdata, where, usrdata);
    break;
  case GRB_CB_MIPSOL:
    return sol_heur(model, cbdata, where, usrdata);
  default:
    return 0;
  }
  return 0;
}


//used in past experiments; currently unused:
int modified = 0;
int k = 0;
int modified_branching(GRBmodel *model, void *cbdata, int where, void *usrdata) {
  if(where != GRB_CB_MIPSOL) return 0;
  if(k++ < 10) return 0;
  int state = 0;
  GRBterminate(model);
  //state = GRBsetintattrelement(model, GRB_INT_ATTR_BRANCHPRIORITY, 30, 1);
  modified = 1;
  k = 0;
  //printf("state = %d\n", state);
  return state;
}


int gurobi_random_bounded(int *branched, int idx_min, int idx_max) {
  int *eligible = CALLOC(idx_max - idx_min + 1, int);
  int eligible_cnt = 0;
  for (int i = idx_min; i <= idx_max; i++) {
    if (!branched[i]) {
      eligible[eligible_cnt++] = i;
    }
  }
  int candidate = eligible_cnt > 0 ? eligible[lrand48() % eligible_cnt] : -1;
  free(eligible);
  return candidate;
}

typedef struct pair_t {
  int key;
  double val;
} pair_t;

int cmppairs(const void *a, const void *b) {
  return ((pair_t *) a)->val > ((pair_t *) b)->val;
}

/* Returns an array containing the num closest samples to the hyperplane
 * This can be done faster (linear time) using order statistics */
int *gurobi_by_violation(env_t *env, double *sol, int num) {
  int *cands = CALLOC(num, int);
  double branch_target = env->params->branch_target;
  double max_frac = -__DBL_MAX__;
  int max_idx = -1;
  samples_t *samples = env->samples;
  int idx_max = violation_idx(0, samples);
  pair_t *pairs = CALLOC(idx_max, pair_t);
  for (int i = samples->dimension+2; i < idx_max; i++) {
    //double value = glp_get_col_prim(p, i);
    double value = sol[i];
    if (index_label(i, samples) > 0) {
      value = 1. - value;
    }
    value = fabs(value - branch_target);
    pairs[i] = (pair_t) {i, value};
  }
  qsort(pairs, idx_max, sizeof(pair_t), cmppairs);
  for(int i = 0; i < num; i++)
    cands[i] = pairs[i].key;
  free(pairs);
  return cands;
}

double *incumb;
int modified_rins(GRBmodel *parent_model, void *cbdata, int where, void *usrdata) {
  env_t *env = usrdata;
  if(where == GRB_CB_MIPSOL) {
    int nvars;
    int state;
    state = GRBgetintattr(parent_model, GRB_INT_ATTR_NUMVARS, &nvars);
    if(k++ == 0)
      incumb = CALLOC(1, double);
    free(incumb);
    incumb = CALLOC(nvars, double);
    state = GRBcbget(cbdata, where, GRB_CB_MIPSOL_SOL, incumb);
    //env->solution_data->integer_solution = CALLOC(nvars, double);
    //for(int i = 0; i < nvars; i++)
    //  env->solution_data->integer_solution[i] = incumb[i];
    /*for(int i = env->samples->dimension + 2; i < nvars; i++) {
      printf("incumb[%d] = %0.3f\n", i, incumb[i]);
      }*/
    //free(incumb);
    return state;
  } if(where == GRB_CB_MIPNODE && k++ % FREQ == 0) {
    int state = 0;
    
    GRBmodel *model = GRBcopymodel(parent_model);
    //GRBsetcallbackfunc(model, test_heur, usrdata);
    GRBsetcallbackfunc(model, NULL, NULL);
    state |= GRBsetintparam(GRBgetenv(model), "RINS", 0);
    GRBupdatemodel(model);
  
    int nvars;
    state |= GRBgetintattr(model, GRB_INT_ATTR_NUMVARS, &nvars);
    printf("Got nvars. State = %d\n", state);
    double *h = CALLOC(nvars, double); //relaxation solution
    state |= GRBcbget(cbdata, where, GRB_CB_MIPNODE_REL, h);
    //printf("Got relaxation. State = %d\n", state);
    if(state == 10005) return 0; //sometimes cannot load relaxation
    double *solution = blank_solution(env->samples);
    hyperplane_to_solution(h, solution, env);
    //double *incumb = env->solution_data->integer_solution;
    /*for(int i = env->samples->dimension + 2; i < nvars; i++) {
      if(incumb[i] != 0 && incumb[i] != 1)
	printf("[HEUR] NON-INTEGER-INCUMBENT: incumb[%d] = %0.3f\n", i, incumb[i]);
	}*/
    int common = 0;
    for(int i = env->samples->dimension + 2; i < nvars; i++) {
      if(incumb[i] == solution[i]) {
	//fix value by setting LB and UB to the same value
	state |= GRBsetdblattrelement(model, "LB", i, incumb[i]);
	state |= GRBsetdblattrelement(model, "UB", i, incumb[i]);
	common++;
      }
    }
    printf("Fixed vars. State = %d\n", state);

    if(common == 0) {
      printf("[HEUR] subproblem identical to original\n");
      return state;
    }

    state |= GRBsetdblattrarray(model, GRB_DBL_ATTR_START, 0, nvars, incumb);
    printf("Set start. State = %d\n", state);
    /*printf("Incumbent:\n");
    for(int i =0; i < nvars; i++) {
      printf("%0.3f%s", incumb[i], (i == nvars - 1) ? "\n" : " ");
      }*/


    state |= GRBsetdblparam(GRBgetenv(model), "NodeLimit", LIM);
    //state |= GRBsetintparam(GRBgetenv(model), "OutputFlag", 0);
    printf("State = %d\n", state);
    printf("[HEUR] STARTING SUBPROBLEM (fixed %d vars):\n", common);

    state |= GRBoptimize(model);
    printf("State = %d\n", state);

    printf("[HEUR] END SUBPROBLEM\n");

    double *new_sol = CALLOC(nvars, double);
    state |= GRBgetdblattrarray(model, "X", 0, nvars, new_sol);
    printf("State = %d\n", state);

    state |= GRBcbsolution(cbdata, new_sol, NULL);
    //  printf("[HEUR] STATE = %d\n", state);
    free(h);
    free(solution);
    //free(incumb);
    free(new_sol);
    GRBfreemodel(model);
    return state;
  }
  return 0;
}

double cb_best_obj = 0;
int cb_n_cones = 0;
int gurobi_cone_callback(GRBmodel *model, void *cbdata, int where, void *usrdata) {
  //whenever we find a new integer solution, expand the cone and add solve within it, then add the constraints to force future solutions to be outside the cone
  if(where != GRB_CB_MIPSOL) return 0;

  int state = 0;
  
  double obj;

  state |= GRBcbget(cbdata, where, GRB_CB_MIPSOL_OBJ, &obj);
  if(obj <= 0) return 0;
  

  int nvars;
  env_t *env = usrdata;
  state |= GRBgetintattr(model, GRB_INT_ATTR_NUMVARS, &nvars);
  double *h = CALLOC(nvars, double); //current best integer soln
  state |= GRBcbget(cbdata, where, GRB_CB_MIPSOL_SOL, h);

  int *cone = expand_cone(env, h);
  if(!cone) return 0;

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
  h = single_gurobi_run(0, 12000, 1200, env, &p);

  state |= add_gurobi_outside_cone_lazy(cbdata, env, cone, cb_n_cones);  
  
  cb_best_obj = fmax(h[0], cb_best_obj);
  cb_n_cones++;
  printf("----------------------------------- STATUS ---------------------------------------\n");
  printf("%d cones found so far\n", cb_n_cones);
  printf("best objective = %g\n", cb_best_obj);
  printf("----------------------------------------------------------------------------------\n");

  free(h);
  free(cone);

  return state;
}

int gurobi_tree_search_callback(GRBmodel *model, void *cbdata, int where, void *usrdata) {
  if(where != GRB_CB_MIPNODE) return 0;
  //generate solution by rounding
  //find which vs are 0, constraining the subspace
  //then generate a random hyperplane in that subspace (or maybe more than 1)
  //finding the subspace is done with QR fac, so gsl will be used

  int state = 0;
  int nvars;
  env_t *env = usrdata;
  state |= GRBgetintattr(model, GRB_INT_ATTR_NUMVARS, &nvars);
  double *rel = CALLOC(nvars, double);
  state |= GRBcbget(cbdata, where, GRB_CB_MIPNODE_REL, rel);
  
  int ncols = 0;
  double tot_sum = 0;
  int start_v = violation_idx(0, env->samples);
  int end_v = nvars - 1;
  for(int i = start_v; i < end_v; i++) {
    if(fabs(rel[i]) <= 1e-10) ncols++;
    tot_sum += rel[i];
  }
  printf("subspace dim = %d, total sum = %g\n", ncols, tot_sum);

  
  free(rel);
  
  return 0;
}
