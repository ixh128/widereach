#include "helper.h"
#include "widereach.h"

#include <math.h>
#include <memory.h>
#include <time.h>

#define TRY_MODEL(premise, step)                                               \
  TRY(premise, error_handle(state, model, step), NULL)

#define MSG_LEN 256

int error_handle(int state, GRBmodel *model, char *step) {
  if (!state) {
    return 0;
  }

  GRBenv *env = GRBgetenv(model);
  char msg[MSG_LEN];
  snprintf(msg, MSG_LEN, "Error (%s): %i\nError message: %s\n", step, state,
           GRBgeterrormsg(env));
  GRBmsg(env, msg);
  return state;
}

double *gurobi_relax(unsigned int *seed, int tm_lim, int tm_lim_tune,
                     env_t *env) {
  /* samples_t *samples = env->samples;
  env->solution_data = solution_data_init(samples_total(samples));

  if (seed != NULL) {
      srand48(*seed);
  }*/

  int state;
  GRBmodel *model;

  TRY_MODEL(model = gurobi_milp(&state, env), "model creation");

  TRY_MODEL(state =
                GRBsetstrparam(GRBgetenv(model), "LogFile", "gurobi_log.log"),
            "set log file");

  TRY_MODEL(state = GRBsetdblparam(GRBgetenv(model), "TuneTimeLimit",
                                   tm_lim_tune / 1000.),
            "set time limit for tuning")

  printf("optimize ...\n");

  double k;
  TRY_MODEL(GRBupdatemodel(model), "update model");
  TRY_MODEL(GRBgetdblattr(model, "Kappa", &k), "get condition number");
  printf("Kappa = %.3e\n", k);

  TRY_MODEL(state =
                GRBsetdblparam(GRBgetenv(model), "TimeLimit", tm_lim / 1000.),
            "set time limit");

  /*GRBwrite(model, "pre.prm"); //save params before tuning

  TRY_MODEL(state = GRBtunemodel(model), "autotune");
  int nresults;
  TRY_MODEL(state = GRBgetintattr(model, "TuneResultCount", &nresults), "get
  tune results"); printf("------------%d results--------------\n", nresults);
  if(nresults > 0)
    TRY_MODEL(state = GRBgettuneresult(model, 0), "apply tuning");
    GRBwrite(model, "post.prm");

    GRBwrite(model, "tmp.lp");*/

  // convert integer vars to double
  int nvars;
  TRY_MODEL(state = GRBgetintattr(model, GRB_INT_ATTR_NUMVARS, &nvars),
            "get number of variables");
  char *conts = CALLOC(nvars, char);
  char *vtypes = CALLOC(nvars, char);
  TRY_MODEL(state = GRBgetcharattrarray(model, "VType", 0, nvars, vtypes),
            "get variable types");
  for (int i = 0; i < nvars; i++)
    conts[i] = GRB_CONTINUOUS;
  TRY_MODEL(state = GRBsetcharattrarray(model, "VType", 0, nvars, conts),
            "set vars to continuous");

  free(conts);

  TRY_MODEL(state = GRBoptimize(model), "optimize");

  int optimstatus;
  TRY_MODEL(state = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &optimstatus),
            "get optimization status")

  // find optimal value of decision vars
  double *result = CALLOC(nvars + 1, double);
  TRY_MODEL(state = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, result),
            "get objective value");
  TRY_MODEL(state =
                GRBgetdblattrarray(model, GRB_DBL_ATTR_X, 0, nvars, result + 1),
            "get decision vars");

  /*printf("Decision vars:\n");
  for(int i = 0; i < nvars+1; i++)
  printf("%0.3f%s", result[i], (i == nvars) ? "\n" : ", ");*/

  // GRBwrite(model, "soln.json");

  TRY_MODEL(state = GRBsetcharattrarray(model, "VType", 0, nvars, vtypes),
            "reset vars to discrete");
  free(vtypes);
  return result;
}

int set_pos_prio(env_t *env, GRBmodel *model) {
  int P = positives(env->samples);
  int *val = CALLOC(P, int);
  for (int i = 0; i < P; i++)
    val[i] = 1;
  return GRBsetintattrarray(model, "BranchPriority",
                            env->samples->dimension + 1, P, val);
}

int force_all_pos(env_t *env, GRBmodel *model) {
  // just testing - force all xs to be 1:
  int P = positives(env->samples);
  double *val = CALLOC(P, double);
  for (int i = 0; i < P; i++)
    val[i] = 1;
  return GRBsetdblattrarray(model, "LB", env->samples->dimension + 1, P, val);
}

int nrels = 0;
int nfeas = 0;

double switch_time = -1;

double *single_gurobi_run(unsigned int *seed, int tm_lim, int tm_lim_tune,
                          env_t *env, gurobi_param *param_setting) {
  samples_t *samples = env->samples;
  env->solution_data = solution_data_init(samples_total(samples));

  if (seed != NULL) {
    printf("Applying seed %u\n", *seed);
    srand48(*seed);
  }

  int state;
  GRBmodel *model;
  int method;
  if (!param_setting) {
    method = 0;
  } else {
    method = param_setting->method;
  }
  switch (method) {
  case 0:
    TRY_MODEL(model = gurobi_milp(&state, env), "model creation");
    break;
  case 1:
    TRY_MODEL(model = gurobi_qp(&state, env), "model creation");
    break;
  case 2:
    TRY_MODEL(model = gurobi_cones_miqp(&state, env), "model creation");
    TRY_MODEL(state =
                  GRBsetcallbackfunc(model, gurobi_tree_search_callback, env),
              "set callback");
    break;
  case 3:
    TRY_MODEL(model = gurobi_milp_unbiased(&state, env), "model creation");
    break;
  case 4:
    TRY_MODEL(model = gurobi_relaxation(&state, env), "model creation");
    break;
  case 5:
    if (!param_setting->init) {
      printf("no initial solution provided\n");
      return NULL;
    }
    TRY_MODEL(model = gurobi_relax_within_small_cone(&state, env,
                                                     param_setting->init),
              "model creation");
    /*TRY_MODEL(state = GRBcomputeIIS(model), "IIS");
      TRY_MODEL(state = GRBwrite(model, "tmp.ilp"), "write out");*/
    break;
  case 6:
    if (!param_setting->cone) {
      printf("no cone provided\n");
      return NULL;
    }
    TRY_MODEL(model =
                  gurobi_relax_within_cone(&state, env, param_setting->cone),
              "model creation");
    break;
  case 7:
    TRY_MODEL(model = gurobi_milp_strict(&state, env), "model creation");
    /*TRY_MODEL(state = GRBsetcallbackfunc(model, gurobi_cone_callback, env),
      "set callback"); TRY_MODEL(state = GRBsetintparam(GRBgetenv(model),
      "LazyConstraints", 1), "allow lazy constraints");*/
    break;
  case 8:
    TRY_MODEL(model = gurobi_relax_within_subspace(&state, env,
                                                   param_setting->ortho.basis,
                                                   param_setting->ortho.n),
              "model creation");
    break;
  case 9:
    TRY_MODEL(model = gurobi_bilinear(&state, env), "model creation");
    break;
  case 10:
    TRY_MODEL(model = gurobi_milp_tgm(&state, env), "model creation");
    break;
  default:
    printf("invalid method\n");
    return NULL;
  }

  TRY_MODEL(state =
                GRBsetstrparam(GRBgetenv(model), "LogFile", "gurobi_log.log"),
            "set log file");

  TRY_MODEL(state = GRBupdatemodel(model), "update model");

  int nvars;
  TRY_MODEL(state = GRBgetintattr(model, GRB_INT_ATTR_NUMVARS, &nvars),
            "get number of variables");

  // TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "ScaleFlag", 3), "set
  // scale flag");

  TRY_MODEL(state = GRBsetdblparam(GRBgetenv(model), "TuneTimeLimit",
                                   tm_lim_tune / 1000.),
            "set time limit for tuning")
  // TRY_MODEL(state = GRBtunemodel(model), "parameter tuning")

  // Cut generation (TODO warning: just goofing around)
  /* GRBsetintparam(GRBgetenv(model), "GomoryPasses", 0);
  GRBsetintparam(GRBgetenv(model), "CoverCuts", 2);
  GRBsetintparam(GRBgetenv(model), "ImpliedCuts", 2);
  GRBsetintparam(GRBgetenv(model), "InfProofCuts", 2); */

  // printf("optimize ...\n");

  double k;
  TRY_MODEL(GRBupdatemodel(model), "update model");
  TRY_MODEL(GRBgetdblattr(model, "Kappa", &k), "get condition number");
  // printf("Kappa = %.3e\n", k);

  TRY_MODEL(state =
                GRBsetdblparam(GRBgetenv(model), "TimeLimit", tm_lim / 1000.),
            "set time limit");

  /*GRBwrite(model, "pre.prm"); //save params before tuning

TRY_MODEL(state = GRBtunemodel(model), "autotune");
int nresults;
TRY_MODEL(state = GRBgetintattr(model, "TuneResultCount", &nresults), "get tune
results"); if(nresults > 0) TRY_MODEL(state = GRBgettuneresult(model, 0), "apply
tuning"); GRBwrite(model, "post.prm");*/
  if (param_setting->init) {
    TRY_MODEL(state = GRBsetdblattrarray(model, GRB_DBL_ATTR_START, 0,
                                         env->samples->dimension,
                                         param_setting->init),
              "set start hyperplane");
  } else if (method == 1 || method == 8) {
    ;
  } else if (0 && method == 2) {
    /*struct rand_proj_res res = best_random_proj(1, env);
    size_t d = env->samples->dimension;
    int n = samples_total(samples);
    int v_start = violation_idx(0, samples);
    double *random_solution_v = CALLOC(n, double);
    for(int j = 0; j < n; j++)
      random_solution_v[j] = 1;
    for(int j = 0; j < d-1; j++) {
      random_solution_v[res.indices[j]] = 0;
    }

    double *h = CALLOC(d, double);
    for(size_t i = 0; i < d; i++) {
      h[i] = gsl_vector_get(res.w, i);
    }
    double *random_solution_hplane = blank_solution(samples);
    double random_objective_value = hyperplane_to_solution(h,
    random_solution_hplane, env); printf("Objective value = %g\n",
    random_objective_value);

    TRY_MODEL(state = GRBsetdblattrarray(model, GRB_DBL_ATTR_START, 0,
    samples_total(samples)+d+1, random_solution_hplane+1), "set start
    hyperplane");
    //TRY_MODEL(state = GRBsetdblattrarray(model, GRB_DBL_ATTR_START,
    env->samples->dimension+1, samples_total(samples), random_solution_hplane +
    env->samples->dimension+2), "set start xs and ys"); TRY_MODEL(state =
    GRBsetdblattrarray(model, GRB_DBL_ATTR_START, v_start, n,
    random_solution_v), "set start vs");*/

    size_t d = env->samples->dimension;
    int n = samples_total(samples);
    env->params->greer_params = (struct greer_params){.method = 2,
                                                      .use_heapq = 0,
                                                      .trunc = 0,
                                                      .trim = 0,
                                                      .max_heapq_size = -1,
                                                      .mcts_ucb_const = 100,
                                                      .beam_width = 10,
                                                      .classify_cuda = 0,
                                                      .obj_code = WRC,
                                                      .no_displace = 1};

    double *h = best_random_hyperplane_unbiased(1, env);
    h = single_greer_run(env, h);
    double *random_solution_hplane = blank_solution(samples);
    double random_objective_value =
        hyperplane_to_solution(h, random_solution_hplane, env);
    printf("Objective value = %g\n", random_objective_value);

    // TRY_MODEL(state = GRBsetdblattrarray(model, GRB_DBL_ATTR_START, 0,
    // samples_total(samples)+d+1, random_solution_hplane+1), "set start
    // hyperplane");
    TRY_MODEL(state = GRBsetdblattrarray(
                  model, GRB_DBL_ATTR_START, env->samples->dimension + 1,
                  samples_total(samples),
                  random_solution_hplane + env->samples->dimension + 2),
              "set start");

  } else if (method == 3 || method == 7) {
    /*env->params->epsilon_positive = 0;
    env->params->epsilon_negative = 0;
    gurobi_param p = {0, 0, 0, GRB_INFINITY, -1, 0.15, -1, 2};
    double *h = single_gurobi_run(seed, 5000, 1200, env, &p);
    printf("Hyperplane: ");
    for(int i = 0; i < env->samples->dimension+1; i++)
    printf("%0.3f%s", h[i], (i == env->samples->dimension) ? "\n" : " ");*/

    double *h = best_random_hyperplane_unbiased(1, env);

    double *random_solution = blank_solution(samples);
    double random_objective_value =
        hyperplane_to_solution(h, random_solution, env);
    printf("Objective value = %0.3f\n", random_objective_value);
    printf("Precision = %lg, reach = %u\n", precision(random_solution, samples),
           reach(random_solution, samples));
    // printf("h: Precision = %lg, reach = %u\n", precision(h, samples),
    // reach(h, samples));

    TRY_MODEL(state = GRBsetdblattrarray(
                  model, GRB_DBL_ATTR_START, env->samples->dimension + 1,
                  samples_total(samples),
                  random_solution + env->samples->dimension + 2),
              "set start");

    // TRY_MODEL(state = GRBsetdblattrarray(model, GRB_DBL_ATTR_START, 0,
    // samples->dimension, h), "set start"); env->params = params_default();
  } else if (method == 10) {
    printf("Generating best of %d hyperplanes\n", env->params->rnd_trials);
    double *h = best_random_hyperplane(1, env);
    // double *h = CALLOC(env->samples->dimension+1, double);
    // double *h = single_exact_run(env);
    printf("Dimension = %lu\n", env->samples->dimension);
    // for(int i = 0; i < env->samples->dimension+1; i++) h[i] /= 100;
    // printf("Hyperplane: %0.3f %0.3f %0.3f %0.3f\n", h[0], h[1], h[2], h[3]);
    printf("Hyperplane: ");
    for (int i = 0; i < env->samples->dimension + 1; i++)
      printf("%0.3f%s", h[i], (i == env->samples->dimension) ? "\n" : " ");
    printf("Done, printing solution:\n");
    double *random_solution = blank_solution(samples);
    double random_objective_value =
        hyperplane_to_solution(h, random_solution, env);
    printf("Objective value = %0.3f\n", random_objective_value);
    printf("Precision = %lg, reach = %u\n", precision(random_solution, samples),
           reach(random_solution, samples));

    TRY_MODEL(state = GRBsetdblattrarray(
                  model, GRB_DBL_ATTR_START, env->samples->dimension + 1,
                  samples_total(samples),
                  random_solution + env->samples->dimension + 2),
              "set start");

    free(h);
    free(random_solution);
  } else {
    /*printf("Generating best of %d hyperplanes\n", env->params->rnd_trials);
    double *h = best_random_hyperplane(1, env);
    // double *h = CALLOC(env->samples->dimension+1, double);
    // double *h = single_exact_run(env);
    printf("Dimension = %lu\n", env->samples->dimension);
    // for(int i = 0; i < env->samples->dimension+1; i++) h[i] /= 100;
    // printf("Hyperplane: %0.3f %0.3f %0.3f %0.3f\n", h[0], h[1], h[2], h[3]);
    printf("Hyperplane: ");
    for (int i = 0; i < env->samples->dimension + 1; i++)
      printf("%0.3f%s", h[i], (i == env->samples->dimension) ? "\n" : " ");
    printf("Done, printing solution:\n");
    double *random_solution = blank_solution(samples);
    double random_objective_value =
        hyperplane_to_solution(h, random_solution, env);
    printf("Objective value = %0.3f\n", random_objective_value);
    printf("Precision = %lg, reach = %u\n", precision(random_solution, samples),
           reach(random_solution, samples));

    TRY_MODEL(state = GRBsetdblattrarray(
                  model, GRB_DBL_ATTR_START, env->samples->dimension + 1,
                  samples_total(samples),
                  random_solution + env->samples->dimension + 2),
              "set start");

    free(h);
    free(random_solution);*/

    // solve TGM for initial solution:
    /*gurobi_param p2;
    memcpy(&p2, param_setting, sizeof(gurobi_param));
    p2.method = 10;
    double *h = single_gurobi_run(seed, tm_lim, tm_lim_tune, env, &p2);
    double *random_solution = blank_solution(samples);
    double random_objective_value =
        hyperplane_to_solution(h + 1, random_solution, env);
    printf("Objective value = %0.3f\n", random_objective_value);
    printf("Precision = %lg, reach = %u\n", precision(random_solution, samples),
           reach(random_solution, samples));

    TRY_MODEL(state = GRBsetdblattrarray(
                  model, GRB_DBL_ATTR_START, env->samples->dimension + 1,
                  samples_total(samples),
                  random_solution + env->samples->dimension + 2),
              "set start");

    free(h);
    free(random_solution);*/
  }

  TRY_MODEL(state = GRBsetcallbackfunc(model, gurobi_callback, env),
            "set callback");

  /*    if(param_setting) {
        TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "Threads",
     param_setting->threads), "set thread limit"); TRY_MODEL(state =
     GRBsetintparam(GRBgetenv(model), "MIPFocus", param_setting->MIPFocus), "set
     focus"); TRY_MODEL(state = GRBsetdblparam(GRBgetenv(model),
     "ImproveStartGap", param_setting->ImproveStartGap), "set start gap");
        TRY_MODEL(state = GRBsetdblparam(GRBgetenv(model), "ImproveStartTime",
     param_setting->ImproveStartTime), "set start time"); TRY_MODEL(state =
     GRBsetintparam(GRBgetenv(model), "VarBranch", param_setting->VarBranch),
     "set branching strategy"); TRY_MODEL(state =
     GRBsetdblparam(GRBgetenv(model), "Heuristics", param_setting->Heuristics),
     "set heuristics"); TRY_MODEL(state = GRBsetintparam(GRBgetenv(model),
     "Cuts", param_setting->Cuts), "set cuts");
        //TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "RINS",
     param_setting->RINS), "set RINS"); if(param_setting->pos_prio == 1) {
          TRY_MODEL(state = set_pos_prio(env, model), "set branch prio");
        }
        if(param_setting->force_pos == 1) {
          TRY_MODEL(state = force_all_pos(env, model), "set branch prio");
        }
      }*/
  GRBwrite(model, "tmp.lp");

  // TRY_MODEL(state = GRBsetdblparam(GRBgetenv(model), "NodeLimit", 0), "set
  // node limit"); TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "Threads",
  // 1), "set thread limit"); TRY_MODEL(state = GRBsetdblparam(GRBgetenv(model),
  // "Heuristics", 0), "disable heuristics"); TRY_MODEL(state =
  // GRBsetintparam(GRBgetenv(model), "Cuts", 0), "disable cutting planes");

  int optimstatus;
  int dimension = env->samples->dimension;
  // int *branched = CALLOC(nvars, int);

  // TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "Threads", 16), "set
  // thread limit"); startHyperplanes(env, model); TRY_MODEL(state =
  // GRBsetcallbackfunc(model, backgroundHyperplanes, env), "add random
  // hyperplane callback");
  TRY_MODEL(state = GRBoptimize(model), "optimize");
  /*TRY_MODEL(state = GRBcomputeIIS(model), "optimize");
    TRY_MODEL(state = GRBwrite(model, "tmp.ilp"), "write");*/
  // printf("nrels = %d, nfeas = %d\n", nrels, nfeas);
  // stopHyperplanes();
  // printf("Done with first optimization\n");

  /*if(switch_time != -1) {
  // if the solver was terminated by the callback, it will have set switch_time,
  so this will run
  // change the heuristics parameter when this happens
    TRY_MODEL(state = GRBsetdblparam(GRBgetenv(model), "Heuristics", 1), "change
  heuristics"); TRY_MODEL(state = GRBsetdblparam(GRBgetenv(model), "TimeLimit",
  tm_lim/1000. - switch_time), "change time limit"); TRY_MODEL(state =
  GRBoptimize(model), "optimize");
    }*/

  // old code for modified branching strategy (didn't work and not necessary -
  // gurobi does not like being interrupted constantly)
  /*do {
    if(modified) {
      double *sol = CALLOC(nvars, double);
      TRY_MODEL(state = GRBgetdblattrarray(model, "X", 0, nvars, sol), "get
    intermediate X"); int num_cands = 50; int *cands = gurobi_by_violation(env,
    sol, num_cands); for(int i = 0; i < num_cands; i++) {
        TRY_MODEL(GRBsetintattrelement(model, GRB_INT_ATTR_BRANCHPRIORITY,
    cands[i], 1), "set branch priority");
      }
      free(cands);
    }
    modified = 0;
    TRY_MODEL(state = GRBoptimize(model), "optimize");
    TRY_MODEL(state = GRBgetintattr(model, GRB_INT_ATTR_STATUS, &optimstatus),
    "get optimization status"); if(optimstatus == GRB_OPTIMAL) break; }
    while(modified);*/

  // find optimal value of decision vars

  // NOTE: modified - should be nvars and not include the objval
  int sols;
  TRY_MODEL(state = GRBgetintattr(model, "SolCount", &sols), "get sol count");
  if (sols == 0) { // no solution was found
    GRBfreeenv(GRBgetenv(model));
    GRBfreemodel(model);
    return NULL;
  }
  double *result = CALLOC(nvars + 1, double);
  TRY_MODEL(state = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, result),
            "get objective value");
  TRY_MODEL(state =
                GRBgetdblattrarray(model, GRB_DBL_ATTR_X, 0, nvars, result + 1),
            "get decision vars");

  /*double *result = CALLOC(nvars, double);
    TRY_MODEL(state = GRBgetdblattrarray(model, GRB_DBL_ATTR_X, 0, nvars,
    result), "get decision vars");*/

  /*printf("Decision vars:\n");
  for(int i = 0; i < nvars; i++)
  printf("%0.3f%s", result[i], (i == nvars - 1) ? "\n" : " ");*/

  GRBwrite(model, "soln.sol");

  GRBfreeenv(GRBgetenv(model));
  GRBfreemodel(model);

  return result;

  /*
  // glp_scale_prob(p, GLP_SF_AUTO);
  glp_simplex(p, NULL);

  glp_iocp *parm = iocp(env);
  parm->tm_lim = tm_lim;
  parm->bt_tech = GLP_BT_DFS;
  // parm->bt_tech = GLP_BT_BLB;
  MFV chooses the largest {x} (e.g., 0.99 in favor of 0.1)
  * It would be similar to branch_target=1 for the positive samples,
  * but the opposite for negative samples
  // parm->br_tech = GLP_BR_LFV;
  glp_intopt(p, parm);
  free(parm);

  double *result = solution_values_mip(p);
   size_t dimension = samples->dimension;
  double *result = CALLOC(dimension + 2, double);
  double *h = hyperplane(p, samples);
  copy_hyperplane(dimension, result, h);
  free(h);
  double obj = result[0] = glp_mip_obj_val(p);
  glp_printf("Objective: %g\n", obj);
  // result[dimension + 1] = obj;

  int index_max = violation_idx(0, env.samples);
  for (int i = 1; i <= index_max; i++) {
      glp_printf("%s:\t%g\n", glp_get_col_name(p, i), glp_mip_col_val(p, i));
  }

  glp_delete_prob(p);
  free(delete_solution_data(env->solution_data));

  // return result; */
  return NULL;
}
double *find_outside_cones(unsigned int *seed, int tm_lim, env_t *env,
                           int **cones, int n_cones) {
  samples_t *samples = env->samples;
  env->solution_data = solution_data_init(samples_total(samples));

  if (seed != NULL) {
    printf("Applying seed %u\n", *seed);
    srand48(*seed);
  }

  int state;
  printf("about to make model\n");
  GRBmodel *model = gurobi_find_outside_cones(&state, env, cones, n_cones);
  if (!model)
    printf("no model\n");

  TRY_MODEL(state =
                GRBsetstrparam(GRBgetenv(model), "LogFile", "gurobi_log.log"),
            "set log file");

  TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "MIPFocus", 1),
            "set focus");

  TRY_MODEL(state = GRBupdatemodel(model), "update model");

  int nvars;
  TRY_MODEL(state = GRBgetintattr(model, GRB_INT_ATTR_NUMVARS, &nvars),
            "get number of variables");

  // TRY_MODEL(state = GRBsetintparam(GRBgetenv(model), "ScaleFlag", 3), "set
  // scale flag");

  printf("optimize ...\n");

  TRY_MODEL(state =
                GRBsetdblparam(GRBgetenv(model), "TimeLimit", tm_lim / 1000.),
            "set time limit");

  TRY_MODEL(state = GRBoptimize(model), "optimize");

  double *result = CALLOC(nvars, double);
  // TRY_MODEL(state = GRBgetdblattr(model, GRB_DBL_ATTR_OBJVAL, result), "get
  // objective value");
  TRY_MODEL(state = GRBgetdblattrarray(model, GRB_DBL_ATTR_X, 0, nvars, result),
            "get decision vars");

  GRBwrite(model, "soln_out.sol");

  GRBfreeenv(GRBgetenv(model));
  GRBfreemodel(model);

  return result;
}
