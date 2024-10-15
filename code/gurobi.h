/* --------------------------- Execution -------------------------------- */

/** Return a Gurobi instance intialized from the given environment */
GRBmodel *gurobi_milp(
  /** Initialization error code (see Gurobi) */
  int *state, 
  /** Instance environment */
  const env_t *);


/** Run gurobi relaxation only

 @return a new array in which 
 the zeroth element is the objective value in the relaxation solution at the end of 
 the run,
 and the next elements are the values of the decision variables. */
double *gurobi_relax(
  /** Seed for the random number generator, or NULL if the drand48 does not
   * need to be reseeded */
  unsigned int *seed, 
  /** Time limit in milliseconds */
  int tm_lim, 
  /** Time limit in milliseconds for tuning the model */
  int tm_lim_tune, 
  env_t *);

typedef struct gurobi_param {
  int threads;
  int MIPFocus;
  double ImproveStartGap;
  double ImproveStartTime;
  int VarBranch;
  double Heuristics;
  int Cuts;
  int RINS;
  int method; //0 for milp, 1 for qp, 2 for cones
  double *init; //optional initial solution (only for relax within cone right now)
  int *cone; //optional cone (only for method = 6)
  struct {
    sample_locator_t *basis;
    int n;
  } ortho; //force hplane to be othogonal to these samples (only for method = 8)
} gurobi_param;

/** Launch a single experiment 
 
 @return a new array in which 
 the zeroth element is the objective value in the MIP solution at the end of 
 the run,
 and the next elements are the values of the decision variables. */
double *single_gurobi_run(
  /** Seed for the random number generator, or NULL if the drand48 does not
   * need to be reseeded */
  unsigned int *seed, 
  /** Time limit in milliseconds */
  int tm_lim, 
  /** Time limit in milliseconds for tuning the model */
  int tm_lim_tune, 
  env_t *,
  gurobi_param *);

/** Random hyperplane callback for gurobi */
int backgroundHyperplanes(GRBmodel *, void *, int, void *);

void startHyperplanes(env_t *, GRBmodel *);
void stopHyperplanes();
void feature_scaling(env_t *);

int gurobi_callback(GRBmodel *, void *, int, void *);

#include <gsl/gsl_matrix.h>

/** Representation of system 7 from the paper */
typedef struct sys7_t {
  gsl_matrix *A;
  gsl_vector *b;
  int *ref; //ref[i] is the index in env->samples corresponding to row i in matrix A
} sys7_t;

double *random_constrained_hyperplane(env_t *env, double *rel_sol, sys7_t *s);

sys7_t *generate_fixedpt_mat(env_t *env, double *rel_sol);

void *compute_inseparabilities(env_t *, int);

int add_gurobi_hyperplane(GRBmodel *, size_t, int);
int add_gurobi_sample_var(GRBmodel *, int, char *);
int error_handle(int state, GRBmodel *model, char *step);
double *single_gurobi_cones_run(unsigned int *, int, int, env_t *);

GRBmodel *gurobi_milp_unbiased(int *, const env_t *);
GRBmodel *gurobi_qp(int *, const env_t *);
GRBmodel *gurobi_cones_miqp(int *, const env_t *);
GRBmodel *gurobi_relaxation(int *, const env_t *);
GRBmodel *gurobi_relax_within_small_cone(int *state, const env_t *env, double *w);
GRBmodel *gurobi_relax_within_cone(int *state, const env_t *env, int *cone);
GRBmodel *gurobi_milp_strict(int *, const env_t *);

GRBmodel *init_gurobi_model(int *state, const env_t *env);
GRBmodel *gurobi_find_outside_cones(int *state, const env_t *env, int **cones, int n_cones);
double *find_outside_cones(unsigned int *seed, int tm_lim, env_t *env, int **cones, int n_cones);
int add_gurobi_outside_cone_lazy(void *cbdata, const env_t *env, int *cone, int cone_idx);

int gurobi_cone_callback(GRBmodel *model, void *cbdata, int where, void *usrdata);
int gurobi_tree_search_callback(GRBmodel *model, void *cbdata, int where, void *usrdata);

GRBmodel *gurobi_relax_within_subspace(int *state, const env_t *env, sample_locator_t *basis, int ss_dim);
GRBmodel *gurobi_bilinear(int *state, const env_t *env);
