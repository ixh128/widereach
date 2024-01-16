/* --------------------------- Samples -------------------------------- */

/** A group of samples */
typedef struct {
	/** Dimension of the sample space, and 
	 * size of the values array of all samples in the array */
	size_t dimension;
	/** Number of sample categories. */
	size_t class_cnt;
	/** Number of samples in each category */
	size_t *count;
	/** Label of the samples in each category */
	int *label;
	/** Sample array. 
	 * The ith array contains count[i] samples in class label[i].
	 * Each sample contains the value along the given dimensions. */
	double ***samples;
} samples_t;


/** Sample locator: specifies a sample in a sample group by class and
 * index within class */
typedef struct {
	/** Class of the index. 
	 * A negative class denotes a non-sample variable */
	int class;
	size_t index;
} sample_locator_t;

/** @brief Delete the element within the given sample array.
 *
 * It is assumed that all the arrays within the sample
 * have been dynamically allocated.
 * No assumption is made about the allocation of the argument itself.
 *
 * @return the samples */
samples_t *delete_samples(samples_t *);

/** Print all samples in the sample set */
void print_samples(samples_t *);

/** Checks whether the given samples are binary: it contains two classes,
 * the first of which labelled -1 and the other +1 */
int is_binary(const samples_t *);

/** Returns the primary value associated with a label. 
 * The primary value is defined as 1 for samples with positive labels
 * and 0 otherwise. */
double primary_value(int label);


/** Return the total number of samples */
size_t samples_total(const samples_t *);

/** Return the number of positive samples in a binary sample set */
size_t positives(const samples_t *);

/** Return the number of negative samples in a binary sample set */
size_t negatives(const samples_t *);

/** Sets the count samples in the given class with the given label
 * to be uniformly distributed over the hypercube */
void set_sample_class(
		samples_t *samples, 
		size_t class, 
		int label, 
		size_t count);

/** Generates random binary samples in the unit square in the given dimension.
 *
 * The drand48(3) functions must have been seeded before invoking this method.
 *
 * @return A newly allocated group of samples
 **/
samples_t *random_samples(
                /** Number of samples */
		size_t count, 
		/** Number of positives samples.
		 * If positives is greater than or equal to count, 
		 * then all points will be positive. */ 
		size_t positives, 
                /** Dimension of the sample space */
		size_t dimension);


/** Returns a new hyperplane defined as a unit vector followed by a bias */
double *random_hyperplane(size_t dimension);

/** Find whether the given sample is on the positive or negative side 
 * of the given hyperplane 
 *
 * @return 1 if the sample is on the positive side, 0 otherwise */
int side(
		/** Locator of the sample of interest */
		sample_locator_t *, 
		/** Sample collection */
		samples_t *, 
		/** Array of size dimension + 1 containing first the 
		 * hyperplane coefficients w1, ... , wd, and then the 
		 * bias c */
		double *hyperplane,
		/** Precision around the hyperplane */
		double precision);

/** Returns the signed distance of the given sample from the given hyperplane.*/
double distance(
		/** Locator of the sample of interest */
		sample_locator_t *, 
		/** Sample collection */
		samples_t *, 
		/** Array of size dimension + 1 containing first the 
		 * hyperplane coefficients w1, ... , wd, and then the 
		 * bias c */
		double *hyperplane,
		/** Precision around the hyperplane */
		double precision);


/** Returns the violation of the given sample from the given hyperplane. 

    The violation is equal to the distance, except that for 
    positive samples it has been multiplied by -1. */
double sample_violation(
		/** Locator of the sample of interest */
		sample_locator_t *, 
		/** Sample collection */
		samples_t *, 
		/** Array of size dimension + 1 containing first the 
		 * hyperplane coefficients w1, ... , wd, and then the 
		 * bias c */
		double *hyperplane,
		/** Precision around the hyperplane */
		double precision);


/** Return the number of samples in the given class that 
 * are on the positive side of the given hyperplane. */
int side_cnt(
        /** Class identifier */
		int class, 
		/** Sample collection */
		samples_t *, 
		/** Array of size dimension + 1 containing first the 
		 * hyperplane coefficients w1, ... , wd, and then the 
		 * bias c */
		double *hyperplane,
		/** Precision around the hyperplane */
		double precision);

/** Performs a reduction on the samples using the given accumulation operator 
 
 @return a code representing success or failure of the reduction
 */
int reduce(
    samples_t *samples,
    /** Initial value of the result */
    void *initial,
    /** Reduction function */
    int (*accumulator)(
      /** Samples */
      samples_t *samples,
      /** Current sample locator */
      sample_locator_t locator,
      /** Current value of the accumulation */
      void *accumulation, 
      /** Auxiliary information */
      void *aux),
    /** Optional auxiliary information to be passed to the accumulator */
    void *aux);


/* ---------------------- Sample clusters --------------------------- */

/** Descriptor of a set of clusters of samples */
typedef struct {
    /** Number of clusters */
    size_t cluster_cnt;
    /** Array with the number of points in each of the cluster_cnt clusters */
    size_t *count;
    /** Shift parameters for the clusters (as in random_point_affine) */
    double *shift;
    /** Side parameters for the clusters (as in random_point_affine) */
    double *side;
    /** Dimension of the sample space */
    size_t dimension;
} clusters_info_t;


/** Fills in a cluster information record for a single cluster
 * with zero shift and unit side 
 *
 * @return the cluster information record
 */
clusters_info_t *clusters_info_singleton(
  clusters_info_t *info, 
  size_t count, 
  size_t dimension);

/** @brief Delete the cluster descriptor.
 *
 * It is assumed that all the arrays within the cluster information
 * have been dynamically allocated.
 * No assumption is made about the allocation of the argument itself.
 *
 * @return the cluster descriptor */
clusters_info_t *delete_clusters_info(clusters_info_t *);

/** @return total number of samples across all counters */
size_t clusters_count(clusters_info_t *);

/** Generates random binary samples. Each sample class is described by the 
 * corresponding entry in the cluster information array.
 *
 * The drand48(3) functions must have been seeded before invoking this method.
 *
 * @return A newly allocated group of samples
 **/
samples_t *random_sample_clusters(
  /** Array of two cluster information records. 
   * The dimension of the samples is the largest of the dimension of the 
   * two records. */
  clusters_info_t *);


/* ---------------------- Read Samples ------------------------------ */
/** Allocate and read a single vector. */
double *read_vector(
    /** Input file */
    FILE *, 
    /** Vector dimension */
    size_t dimension);

/** Read all samples from the given class. */
void read_class(
    /** Input file */
    FILE *,
    /** Samples. 
     * count[class] is assumed to have been initialized to the
     * number of samples in the class.
     * The samples[class] field will contain the samples from the
     * file. */
    samples_t *samples, 
    /** Class to be read */
    int class);

/** Read all samples */
void read_classes(
    /** Input file */
    FILE *infile, 
    /** Samples. 
     * count[class] is assumed to have been initialized to the
     * number of samples in the class.
     * The samples field will contain the samples from the file. */
    samples_t *samples);

/** Allocate and read binary samples from the given file */
samples_t *read_binary_samples(FILE *);


/* ---------------------- Simplexes --------------------------------- */

/** Descriptor of a set of simplex clusters */
typedef struct {
  /** Total number of samples */
  size_t count;
  /** Number of positive samples */
  size_t positives; 
  /** Number of clusters among which the positive samples are partitioned
   * (currently supported: 1, 2) */
  size_t cluster_cnt;
  /** Dimension of the sample space */
  size_t dimension;
  /** Simplex side length */
  double side;
  /** Number of points in each simplex, or NULL for half of the positives to be divided evenly among the clusters. */
  size_t *cluster_sizes;
} simplex_info_t;

/** Generates random binary samples in the simplex in the given dimension.
 *
 * The drand48(3) functions must have been seeded before invoking this method.
 *
 * @return A newly allocated group of samples
 **/
samples_t *random_simplex_samples(simplex_info_t *);


/* ---------------------- Sparse Vectors ---------------------------- */

/** Sparse vector in GLPK format */
typedef struct {
	/** Length of the significant part of ind and val */
	int len;
	/** Additional space at the end of ind and val */
	size_t extra;
	/** Index vector */
	int *ind;
	/** Value vector */
	double *val;
} sparse_vector_t;


/** Allocates a new sparse vector of zero len and the given extras. */
sparse_vector_t *sparse_vector_blank(size_t extra);

/** Deallocates the ind and val vectors in a sparse vector, and 
 * sets the length and extra to zero.
 * It assumes that ind and val were dynamically allocated.
 * 
 * @return the sparse vector */
sparse_vector_t *delete_sparse_vector(sparse_vector_t *);

/** Convert an array of doubles into a new sparse vector. */
sparse_vector_t *to_sparse(
                /** Length of the double array */
		size_t nmemb, 
		double *,
		/** Extra len to be left at the end of the sparse vector */
		size_t extra);

/** Multiple a sparse vector by a scalar.
 *
 * @return The sparse vector after multiplication. */
sparse_vector_t *multiply(sparse_vector_t *, double);

/** Appends an element at the end of the sparse vector. 
 *
 * @return -1 if no extra space is available 
 * (and thus the item was not added) or 
 * the amount of extra space after the insertion. */
int append(sparse_vector_t *, 
	/** Index of the new element.
        It is assumed that this index has not been previously defined. */
	int ind, 
	/** Value to be inserted */
	double val);


/** Creates and returns a sparse vector obtained by retaining only the 
    elements that satisfy the given filter. It assumes that the vector
    is not null. */
sparse_vector_t *filter(
    /** Sparse vector to filter */
    sparse_vector_t *v, 
    /** Predicate that denotes which elements of the sparse vector will be
     * retained. Its argument must be a sparse vector, an index into the 
     * sparse vector, and a pointer to optional information that can be used
     * by the predicate. */
    int (*predicate)(sparse_vector_t *, int, void *),
    /** Additional optional information for potential use by the predicate */
    void *info);


/* --------------------- Parameters                 ---------------------- */

/** Heuristic method */
typedef
enum {
	/** Round the fractional variable's value to the 
	 * closest feasible integer value. */
	simple, 
	/** Round to the side of hyperplane */
	deep 
} iheur_method_t;


/** Problem instance parameters */
typedef struct {
	/** Problem name */
	char *name;
	/** Verbosity level (GLPK enum) */
	int verbosity;
	/** Precision threshold */
	double theta;
	/** Tolerance for positive samples */
	double epsilon_positive;
	/** Tolerance for negative samples */
	double epsilon_negative;
	/** Tolerance for the precision constraint */
	double epsilon_precision;
	/** Lagrangian multiplier of the precision constraint */
	double lambda;
	/** Violation bound: 
     *  either 0 for unbounded variables (GLP_FR)
     *  or 1 for variable with lower bound (GLP_LO) */
	int violation_type;
	/** Target value of the branching variable */
	double branch_target;
	/** Method to use to round to an integer solution */
	iheur_method_t iheur_method;
    /** Number of initial random trials to find an integer solution */
    int rnd_trials;
    /** Number of random trials at each invocation of iheur */
    int rnd_trials_cont;
    /** 1 if this is for a continuous program (xs and ys should be continuous) */
    int cont;
    /** epsilon for the quadratic program */
    double epsilon_prob;
} params_t;


/** Return a new parameter set with default values */
params_t *params_default();


/* ---------------------- Label conversions  --------------------------- */

/** Convert a sample label to its coefficient in the objective function */
double label_to_obj(int label);

/** Convert a sample label to the prefix of the corresponding variable name */
char label_to_varname(int label);

/** Convert a sample label to 
 *  the right hand side of the corresponding constraint */
double label_to_bound(int label, params_t *);


/* ---------------------------- Solution Data -------------------------- */

/** Data updated by the callback during the solution of the program */
typedef struct {
	/** Number of elements in rank that are significant */
	size_t rank_significant;
	/** Sorted list of indexes in order of branching priority */
	int *rank;
    /** Last node that branched */
    int branching_node;
    /** GLPK indexes sorted by hyperplane violation */
    int *violation_index;
    /** Best integer solution found by iheur */
    double *integer_solution;
    /** Objective value achieved by the best integer solution */
    double intopt;
} solution_data_t;


/** Allocate and return a new solution data of the given size with no
 * significant element. */
solution_data_t *solution_data_init(size_t);

/** Deallocate and return the elements in the solution data, 
 * but not the solution data. */
solution_data_t *delete_solution_data(solution_data_t *);

/** Append the given index as the last significant element of the solution
 * data */
solution_data_t *append_data(solution_data_t *, int index);

/** Update the integer solution and its objective value with the provided
 * arguments */
void update_solution(
        solution_data_t *,
        /** New solution vector */
        double *solution, 
        /** Objective value of the new solution vector */
        double value,
        /** Indexes sorted by violation */
        int *violation_index);

/* ---------------------------- Environment ----------------------------- */

/** Environment */
typedef struct {
	/** Sample set */
	samples_t *samples;
	/** Problem instance parameters */
	params_t *params;
	/** Solution data */
	solution_data_t *solution_data;
} env_t;

/** Deallocate the samples and the parameters.
 *
 * It assumes that samples and parameters were dynamically allocated */
env_t *delete_env(env_t *);


/* ------------------------ Indexing ---------------------------------- */

/*
 * Maps addressing in a binary classification problem instance (env) into 
 * indexing in a GLPK problem (glp_prob) according to the follwoing convetions.
 *
 * *Variables (columns)*
 * - 1 to dimension w
 * - dimension+1 c
 * - dimension+2 to dimension+positives+1 xi
 * - dimension+positives+2 to dimension+samples+1 yj
 * - dimension+samples+2 V
 *
 * *Constrains (rows)*
 * - 1 to positives xi
 * - positives+1 to samples yj
 * - samples+1 V 
 */

/** Return the index corresponding to a sample */
int idx(
	/** 0: return column index, 1: row index */
	int direction, 
	/** Sample class */
    int class, 
	/** Sample index within class */
	size_t sample_index, 
	/** Sample collection */
	samples_t *);


/** Maps the index of a decision variable into a sample locator.
 *
 * In case of error (e.g., the index does not belong to a sample)
 * it returns class = -1. */
sample_locator_t *locator(
		/** Decision variable index */
		int index, 
		/** Sample set */
		samples_t *);

/** Appends an element at the end of the sparse vector. 
 *
 * @return -1 if no extra space is available 
 * (and thus the item was not added) or 
 * the amount of extra space after the insertion. */
int append_locator(sparse_vector_t *, 
	/** Locator of the new element.
        It is assumed that this index has not been previously defined. */
	sample_locator_t *, 
	/** Value to be inserted */
	double val,
    samples_t*);

/** Prints the sample referenced by the locator */
void print_sample(sample_locator_t, samples_t *);

/** Return the minimum or maximum index corresponding to an index of a 
 * given class */
int idx_extreme(
	/** 0: return column index, 1: row index */
	int direction, 
	/** Sample class */
    int class, 
	/** 0: return the minimum index, 1: max */
	int extreme,
	/** Sample collection */
	samples_t *);

/** Return the index corresponding to the violation variable */
int violation_idx(
	/** 0: return column index, 1: row index */
	int direction, 
	/** Sample collection */
	samples_t *);

/** Compute the label of the given variable */
int index_label(int i, samples_t *);

/** Fills in a sparse vector in which all decision variables of the given
 * class are assigned the given coefficient 
 *
 * @return the sparse vector */
sparse_vector_t *cover_row(
		/** The sparse vector to be filled */
                sparse_vector_t *,
		/** Class of the decision variables */
                size_t class,
		/** Coefficient of the decision variables */
                double coef,
		/** Sample set */
                samples_t *);

/** Returns a new sparse vector representing the left hand side of the 
 * precision constraint */
sparse_vector_t *precision_row(
		/** Sample set */
		samples_t *, 
		/** Precision threshold */
		double theta);

/** Returns a new sparse vector representing the left hand side of the 
 * precision constraint, for thresholded vars */
sparse_vector_t *precision_row_thr(
		/** Sample set */
		samples_t *, 
		/** Precision threshold */
		double theta);


/** Finds and return the class of the given GLPK decision variable */
int index_to_class(int idx, samples_t *);

/** Create a new solution in which all elements are 1/2 */
double *blank_solution(samples_t *);

/** Convert a hyperplane into an vector of sample distances from the 
 * hyperplane. 
    WARNING: currently unsupported and utested */
void hyperplane_to_distance(
    /** The hyperplane around which the integer solution is to be computed */
    double *hyperplane, 
    /** The distance vector. If the hyperplane is NULL, it is left unchanged. */
    double *distance,
    env_t *);

/** Convert a hyperplane into an integer solution vector 
 * 
 * @return the objective value achieved by the integer solution, or 
            -DBL_MAX if the hyperplane is NULL. */
double hyperplane_to_solution(
    /** The hyperplane around which the integer solution is to be computed */
    double *hyperplane, 
    /** The solution vector, or NULL if the solution is not needed.
     Note that if solution is not null and solution[i] is integer,
     its value is not replaced by the hyperplane side. 
     If the hyperplane is NULL, the solution is left unchanged. 
     Solution indexing is in GLPK style, with solution[0] unused. */
    double *solution,
    env_t *);

/** Same as hyperplane_to_solution, but takes only the parameters and samples
 * rather than the full environment. */
double hyperplane_to_solution_parts(
  double *, double *, params_t *, samples_t *);

/** @return reach of the given integer solution for the given samples */
unsigned int reach(double *solution, samples_t *);

/** @return number of false positives in the given integer solution 
 *for the given samples */
unsigned int false_positives(double *solution, samples_t *);

/** @return precision of the given integer solution for the given samples */
double precision(double *solution, samples_t *);

/** Returns an array of GLPK indexes in increasing order of violation
 * from the given hyperplane */
int *sorted_by_violation(double *hyperplane, env_t *);


/* --------------------------- Random number generation ------------------ */

/** Return the square of the length of the 
 * d-dimensional vector given as argument */
double length_squared(
    /** Vector dimention */
    size_t d, 
    /** Vector */
    double *w);

/** Generates two normal random variables and stores the result in the
 * vector passed as an argument. */
void normal_pair(double *w);

/** Multiply a vector by a scalar. */
double multiply_scalar(
    /** Scalar factor */
    double a, 
    /** Size of the vector to be multiplied */
    size_t d, 
    /** Vector to be multiplied */
    double *w);

/** Generate a random point in the unit hypercube of the given dimension.
 * It returns a newly allocated vector. */
double *random_point(size_t dimension);

/** Generate a random point in the hypercube of the give dimension 
 * located from [shift .... shift]^T to [shift+side ... shift+side]^T 
 * (effectively translating the unit hypercube so that the lower left corner
 * is shift away in every direction and scaling it so that each side has 
 * the given length.)
 */
double *random_point_affine(size_t dimension, double shift, double side);

/** Generates a random unit d-dimensional vector. */
void random_unit_vector(
    /** Vector size */
    size_t d, 
    /** Random unit d-dimensional vector, 
     *  stores the result of the computation */
    double *w);

/** Generates a random vector in the volume under the simplex of the given
 * dimension. It returns a newly allocated vector.
 */
double *random_simplex_point(
    /** Simplex side */
    double side,
    /** Vector size */
    size_t dimension);

/** Copy a hyperplane of the given dimension into another one.
 * A hyperplane is defined as dimension+1 vector in which the first
 * dimension component contain a vector perpendicular to the hyperplane
 * and the last component contains the intercept. */
void copy_hyperplane(size_t dimension, double *dest, double *src);

/** Generates a random hyperplane through a random point on the unit 
 * hypersquare. The orthogonal vector part of the hyperplane as unit size.
   Returns a newly allocated hyperplane.  */
double *random_hyperplane(size_t dimension);

/** Generates rnd_trials hyperplanes and returns 
 * one that achieves the highest value of the objective function. 
   The returned vector has been dynamically allocated. 
   If rnd_trials is zero, then NULL is returned. */
double *best_random_hyperplane(
  /** A Boolean variable denoting whether this is the initial (1) or
    * continuing (0) random hyperplane */
  int initial, 
  env_t *);

/* --------------------------- Execution -------------------------------- */

/** Launch a single experiment 
 
 @return a new array in which 
 the zeroth element is the objective value in the MIP solution at the end of 
 the run,
 and the next elements are the values of the decision variables. */
double *single_run(
  /** Seed for the random number generator, or NULL if the drand48 does not
   * need to be reseeded */
  unsigned int *seed, 
  /** Time limit in milliseconds */
  int tm_lim, 
  env_t *);

/** Binary search on theta to find the approximately largest value of 
 * feasible theta and sets it in the env.
 * 
 * @return the theta threshold
 */
double precision_threshold(
  /** Seed, as in single run */
  unsigned int *seed, 
  env_t *);

/** Scan through all values of theta to find the largest objective value 
 * 
 * @return the theta threshold
 */
double precision_scan(
  /** Seed, as in single run */
  unsigned int *seed, 
  env_t *);

/** Solve by simulated annealing */
double *single_siman_run(unsigned int *seed, int iter_lim, env_t *env, double *h0);

#include <gsl/gsl_blas.h>

/** Stores samples in GSL vector form */
typedef struct {
  /** Dimension of the sample space, and 
   * size of the values array of all samples in the array */
  size_t dimension;
  /** Number of sample categories. */
  size_t class_cnt;
  /** Number of samples in each category */
  size_t *count;
  /** Label of the samples in each category */
  int *label;
  /** Sample array. 
   * The ith array contains count[i] samples in class label[i].
   * Each sample is a pointer to a gsl_vector of its components.*/
  gsl_vector ***samples;
} vsamples_t;

vsamples_t *samples_to_vec(samples_t *);
void vsamples_free(vsamples_t *);

typedef struct round_pts_t {
  size_t n; //number of points
  gsl_vector **pts; //points
  int *npos; //# pos pts attributed to that point
  int *nneg; //# neg pts attributed to that point
} round_pts_t;

typedef struct obj_result {
  double obj;
  double prec;
} obj_result;
obj_result compute_obj(env_t *env, vsamples_t *vs, gsl_vector *w, gsl_matrix *c, round_pts_t *round_pts);
gsl_vector *apply_proj(const gsl_matrix *c, const gsl_vector *x);

/** Solve by cones algorithm */
double *single_exact_run(env_t *env, int n_round_pts);

/** Add an extra feature to each sample which is always equal to 1 */
void add_bias(samples_t *);

/** Scale all samples to have unit norm.
    This yields an equivalent problem if the problem is unbiased */
void normalize_samples(samples_t *);

typedef struct {
  double obj;
  gsl_vector *w;
  int *need_recon; //only for mixed
} cones_result_t; //stores the result of the cones algorithm, if it exists

/** Stores a single subproblem for the cones algorithm */
typedef struct {
  size_t D;
  gsl_matrix *c;
  double score;
  int **m; //this can be ignored for the closed problem
  int *need_recon; //at index D, equals 1 if w = y + au, or -1 if w = y, or 0 if not yet considered, when reconstructing the solution to the D-dimensional subproblem. only for mixed
  struct last_proj {
    size_t class;
    size_t idx; //this is also used as the index of the rounding point
  } last_proj;
} subproblem_t;

/** Priority queue of subproblems */
typedef struct {
  size_t k; //size of heap = 2^k - 1
  size_t sz; //current max size of heap
  size_t n; //current number of elements
  subproblem_t **heap;
  int mt; //1 if multithreaded
  int running; //1 if thread is currently running
  pthread_mutex_t lock; //mutex (if multithreaded)
  pthread_mutex_t running_lock;
  pthread_t thread;
} prio_queue_t;

prio_queue_t *create_prio_queue(int mt);
void prio_enqueue(prio_queue_t *, subproblem_t *);
subproblem_t *prio_pop_max(prio_queue_t *);
size_t prio_get_n(prio_queue_t *);

double *single_siman_cones_run(unsigned int *, int, env_t *, int *);
void add_new_proj_ip(gsl_matrix *, const gsl_vector *);
gsl_vector *get_proj_basis(env_t *, gsl_vector **);
gsl_vector *get_proj_basis_idxs(env_t *, vsamples_t *, int *idxs);

double *best_random_hyperplane_proj(int, env_t *);

double *single_exact_run_open(env_t *);

gsl_vector **uniform_sphere_points(size_t n, size_t d);
double *best_random_hyperplane_unbiased(int, env_t *);
round_pts_t random_round_points(env_t *, vsamples_t *, size_t);
vsamples_t *create_rounded_vs(vsamples_t *, round_pts_t *);

typedef struct {
  round_pts_t round_pts;
  vsamples_t *rvs;
} rounded_pts_t;

rounded_pts_t random_rounded_points(env_t *, vsamples_t *, size_t);
rounded_pts_t cluster_rounded_points(env_t *, vsamples_t *, size_t);
rounded_pts_t cluster_rounded_points_consistent(env_t *, vsamples_t *);
rounded_pts_t cluster_rounded_points_sep_classes(env_t *, vsamples_t *, size_t);

double insep_score(env_t *env);
double insep_score_euclid(env_t *env);

struct rand_proj_res {
  gsl_vector *w;
  gsl_matrix *c;
  int *indices;
} best_random_proj(int initial, env_t *env);
