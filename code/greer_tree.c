#include "helper.h"
#include "widereach.h"

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <sys/time.h>
#include <time.h>

#include <gsl/gsl_linalg.h>
#include <gsl/gsl_rng.h>

#define DEBUG_GR 0

// objectives
int tree_nodes = 0;

params_t *envparams;

tnodestack *create_stack() {
  tnodestack *out = CALLOC(1, tnodestack);
  out->max = 16;
  out->nodes = CALLOC(out->max, tnode *);
  out->top = -1;
  return out;
}

size_t stack_get_n(tnodestack *s) {
  // returns number of elements currently in s
  return s->top + 1;
}

void stack_push(tnodestack *s, tnode *node) {
  if (s->top >= s->max - 1) {
    // if max is reached, double the size
    s->nodes = realloc(s->nodes, 2 * s->max * sizeof(tnode *));
    if (!s) {
      printf("failed to realloc stack, capping at %d\n", s->max);
    } else {
      s->max *= 2;
    }
  }
  s->nodes[++s->top] = node;
}

tnode *stack_pop(tnodestack *s) { return s->nodes[s->top--]; }

int stack_empty(tnodestack *s) { return s->top == -1; }

tnodestack *stack_free(tnodestack *s) {
  free(s->nodes);
  return s;
}

hplane_list *create_list() {
  hplane_list *l =
      CALLOC(1, hplane_list); // this will set everything to 0/NULL already
  return l;
}

lnode *make_lnode(hplane_data *item) {
  lnode *out = CALLOC(1, lnode);
  out->data = item;
  return out;
}

void list_add(hplane_list *l, hplane_data *item) {
  lnode *node = make_lnode(item);
  if (!l->head) {
    l->head = node;
  } else {
    node->next = l->head;
    assert(l->head->prev == NULL);
    l->head->prev = node;
    l->head = node;
  }
  l->poison = 1;
  l->count++;

  node->data->lists++;
}

void free_class_res(class_res *res) {
  free(res->tpos);
  free(res->fpos);
  free(res->tneg);
  free(res->fneg);
  free(res->pz);
  free(res->nz);
  res->tpos = res->fpos = res->tneg = res->fneg = res->pz = res->nz = NULL;
}

void try_free_class_res(class_res *res) {
  // if class res exists, free it
  if (res->tpos) {
    free_class_res(res);
  }
}

void try_free_data(hplane_data *data) {
  // frees the hplane data if extra and res_ent allow it
  if (data->extra) {
    gsl_vector_free(data->v);
    if (!data->res_ent)
      free_class_res(&data->res);
    free(data);
  }
}

void hplane_data_memcpy(hplane_data *dst, hplane_data *src) {
  /*  gsl_vector_memcpy(dst->v, src->v);
  dst->res.ntpos = src->res.ntpos;
  dst->res.nfpos = src->res.nfpos;
  dst->res.npz = src->res.npz;
  dst->res.ntneg = src->res.ntneg;
  dst->res.nfneg = src->res.nfneg;
  dst->res.nnz = src->res.nnz;
  dst->res.obj = src->res.obj;
  dst->res.max_poss_obj = src->res.max_poss_obj;
  dst->res.nwrong = src->res.nwrong;
  dst->res.nright = src->res.nright;
  memcpy(dst->res.tpos, src->res.tpos, dst->res.ntpos*sizeof(int));
  memcpy(dst->res.fpos, src->res.fpos, dst->res.nfpos*sizeof(int));
  printf("ntneg = %d\n", dst->res.ntneg);
  printf("last = %d\n", src->res.tneg[src->res.ntneg-1]);
  memcpy(dst->res.tneg, src->res.tneg, dst->res.ntneg*sizeof(int));
  memcpy(dst->res.fneg, src->res.fneg, dst->res.nfneg*sizeof(int));
  memcpy(dst->res.pz, src->res.pz, dst->res.npz*sizeof(int));
  memcpy(dst->res.nz, src->res.nz, dst->res.nnz*sizeof(int));*/
  try_free_data(dst);
  dst = dup_hplane_data(src);
}

hplane_data *dup_hplane_data(hplane_data *data) {
  hplane_data *cp = CALLOC(1, hplane_data);
  cp->v = gsl_vector_alloc(data->v->size);
  gsl_vector_memcpy(cp->v, data->v);
  memcpy(&cp->res, &data->res, sizeof(class_res));
  cp->res.tpos = CALLOC(cp->res.ntpos, int);
  cp->res.fpos = CALLOC(cp->res.nfpos, int);
  cp->res.tneg = CALLOC(cp->res.ntneg, int);
  cp->res.fneg = CALLOC(cp->res.nfneg, int);
  cp->res.pz = CALLOC(cp->res.npz, int);
  cp->res.nz = CALLOC(cp->res.nnz, int);
  memcpy(cp->res.tpos, data->res.tpos, cp->res.ntpos * sizeof(int));
  memcpy(cp->res.fpos, data->res.fpos, cp->res.nfpos * sizeof(int));
  memcpy(cp->res.tneg, data->res.tneg, cp->res.ntneg * sizeof(int));
  memcpy(cp->res.fneg, data->res.fneg, cp->res.nfneg * sizeof(int));
  memcpy(cp->res.pz, data->res.pz, cp->res.npz * sizeof(int));
  memcpy(cp->res.nz, data->res.nz, cp->res.nnz * sizeof(int));
  cp->extra = 1;
  cp->res_ent = 0;

  return cp;
}

void list_add_copy(hplane_list *l, hplane_data *item) {
  list_add(l, dup_hplane_data(item));
}
void print_vec(gsl_vector *);

hplane_list *list_free(hplane_list *l) {
  lnode *curr = l->head;
  lnode *prev;
  while (curr) {
    prev = curr;
    curr = curr->next;
    prev->data->lists--;
    try_free_data(prev->data);
    free(prev);
  }
  return l;
}

void list_remove_all(hplane_list *l) {
  // removes everything from l
  list_free(l);
  l->head = NULL;
  l->count = 0;
  l->poison = 1;
}

int **blank_branching_data(vsamples_t *vs) {
  // returns a branching_data array which is all zeros
  int **branching_data = CALLOC(2, int *);
  branching_data[0] = CALLOC(vs->count[0], int);
  branching_data[1] = CALLOC(vs->count[1], int);
  return branching_data;
}

int **dup_branching_data(vsamples_t *vs, int **branching_data) {
  // returns a duplicate of branching_data
  if (!branching_data)
    return NULL;
  int **out = blank_branching_data(vs);
  memcpy(out[0], branching_data[0], (vs->count[0]) * sizeof(int));
  memcpy(out[1], branching_data[1], (vs->count[1]) * sizeof(int));

  return out;
}

void free_branching_data(int **branching_data) {
  if (!branching_data)
    return;
  free(branching_data[0]);
  free(branching_data[1]);
  free(branching_data);
}

void tree_remove_node(tnode *node) {
  // removes a node from the tree. frees nothing
  if (node->l_sib) {
    node->l_sib->r_sib = node->r_sib;
  } else {
    // then the node must be the left child or the root
    if (node->parent) {
      assert(node->parent->l_child == node);
      node->parent->l_child = node->r_sib;
    }
  }
  if (node->r_sib) {
    node->r_sib->l_sib = node->l_sib;
  } else {
    // then the node must be the right child or the root
    if (node->parent) {
      assert(node->parent->r_child == node);
      node->parent->r_child = node->l_sib;
    }
  }
  if (node->parent) {
    node->parent->n_children--;
    if (node->explored)
      node->parent->children_explored--;
  }
}

hplane_data *make_hplane_data(gsl_vector *v, class_res res) {
  hplane_data *out = CALLOC(1, hplane_data);
  out->v = v;
  out->res = res;
  return out;
}

void print_vec(gsl_vector *x) {
  printf("[");
  for (int i = 0; i < x->size; i++) {
    printf("%g%s", gsl_vector_get(x, i), i == x->size - 1 ? "]\n" : " ");
  }
}

void print_hplane_list(hplane_list *l) {
  if (!l)
    return;
  lnode *curr = l->head;
  int i = 0;
  while (curr) {
    printf("entry %d: ", i++);
    print_vec(curr->data->v);
    printf("  obj = %g\n", curr->data->res.obj);
    curr = curr->next;
  }
}

void compute_res_obj(env_t *env, class_res *res) {
  params_t *params = env->params;
  if (params->greer_params.obj_code == WRC) {
    double violation =
        (params->theta - 1) * res->ntpos + params->theta * res->nfpos;

    /*if(violation < 0) {
      res->obj = res->ntpos;
      } else {
      res->obj = -1; //infeas <=> neg objective
      }*/
    if (violation < 0)
      violation = 0;

    res->obj = res->ntpos - violation * params->lambda;

    // to compute max possible objective:
    int max_ntpos = res->ntpos + res->npz;
    double min_viol =
        (params->theta - 1) * res->ntpos + params->theta * res->nfpos;
    if (min_viol < 0)
      res->max_poss_obj = max_ntpos;
    else {
      // res->max_poss_obj = -1;
      res->max_poss_obj = max_ntpos - min_viol * params->lambda;
    }
  } else if (params->greer_params.obj_code == MAX_ACC) {
    res->obj = res->ntpos + res->ntneg;
    int max_ntpos = res->ntpos + res->npz;
    int max_ntneg = res->ntneg + res->nnz;
    res->max_poss_obj = max_ntpos + max_ntneg;
  } else if (params->greer_params.obj_code == PREC) {
    res->obj = res->ntpos / ((double)res->ntpos + res->nfpos);
    int max_ntpos = res->ntpos + res->npz;
    res->max_poss_obj =
        max_ntpos /
        ((double)max_ntpos + res->nfpos); // every false positive will remain
                                          // so, so min_nfpos = nfpos
  } else {
    printf("invalid objective code %d\n", params->greer_params.obj_code);
  }
}

class_res classify(env_t *env, gsl_vector *w) {
  vsamples_t *vs = env->vsamples;
  // return classify_cuda(vs, w);
  // classify the samples according to w
  // this could be made faster by a constant factor by not repeating the loops,
  // but then tpos, etc would need to be defined as vectors instead of arrays
  assert(vs->class_cnt == 2);
  int *pos_res = CALLOC(vs->count[1], int);
  int *neg_res = CALLOC(vs->count[0], int);
  class_res out = {0};

  if (DEBUG_GR) {
    printf("classifying w = ");
    print_vec(w);
  }
  for (size_t i = 0; i < vs->count[1]; i++) {
    // positive samples
    double x;
    gsl_blas_ddot(vs->samples[1][i], w, &x);
    if (fabs(x) < 1e-10) {
      //      printf("positive-class zero %ld\n", i);
      out.npz++;
      pos_res[i] = 0;
    } else if (x > 0) {
      // printf("true positive %ld\n", i);
      out.ntpos++;
      pos_res[i] = 1;
    } else {
      // printf("false negative %ld, w . s = %g\n", i, x);
      out.nfneg++;
      pos_res[i] = -1;
    }
  }
  for (size_t i = 0; i < vs->count[0]; i++) {
    // negative samples
    double x;
    gsl_blas_ddot(vs->samples[0][i], w, &x);
    if (fabs(x) < 1e-10) {
      // printf("negative-class zero %ld\n", i);
      out.nnz++;
      neg_res[i] = 0;
    } else if (x > 0) {
      // printf("false positive %ld\n", i);
      out.nfpos++;
      neg_res[i] = 1;
    } else {
      // printf("true negative %ld\n", i);
      out.ntneg++;
      neg_res[i] = -1;
    }
  }

  if (DEBUG_GR)
    printf("results: tpos = %d, fpos = %d, tneg = %d, fneg = %d, pz = %d, nz = "
           "%d\n",
           out.ntpos, out.nfpos, out.ntneg, out.nfneg, out.npz, out.nnz);

  out.tpos = CALLOC(out.ntpos, int);
  int ktpos = 0;
  out.fpos = CALLOC(out.nfpos, int);
  int kfpos = 0;
  out.tneg = CALLOC(out.ntneg, int);
  int ktneg = 0;
  out.fneg = CALLOC(out.nfneg, int);
  int kfneg = 0;
  out.pz = CALLOC(out.npz, int);
  int kpz = 0;
  out.nz = CALLOC(out.nnz, int);
  int knz = 0;

  for (int i = 0; i < vs->count[1]; i++) {
    if (pos_res[i] == 1)
      out.tpos[ktpos++] = i;
    else if (pos_res[i] == -1)
      out.fneg[kfneg++] = i;
    else
      out.pz[kpz++] = i;
  }
  for (int i = 0; i < vs->count[0]; i++) {
    if (neg_res[i] == 1) {
      out.fpos[kfpos++] = i;
    } else if (neg_res[i] == -1)
      out.tneg[ktneg++] = i;
    else
      out.nz[knz++] = i;
  }

  free(pos_res);
  free(neg_res);

  out.nwrong = out.nfneg + out.nfpos;
  out.nright = out.ntpos + out.ntneg;

  compute_res_obj(env, &out);

  return out;
}

tnode *free_subtree(tnode *root) {
  // frees the subtree starting at node root, including all associated vectors
  // (but not samples), not including the root node itself (since root is
  // stack-alloc'd) recursive post-order traversal
  static int nodes_freed = 0;
  static int nodes_not_freed = 0;
  // assert(!root->in_q); //this no longer works as a check, since root may
  // always be in the heapq if we're using a stack
  tnode *next;
  while (root->l_child) {
    next = root->l_child->r_sib;
    free(free_subtree(root->l_child));
    root->l_child = next;
  }
  // then free root itself
  if (root->data->lists <= 0) {
    /*if(root->in_q) {
      printf("attempting to free node in q\n");
      exit(1);
    }*/
    if (root->data->lists < 0)
      printf("error counting lists\n");
    gsl_vector_free(root->data->v);
    try_free_class_res(&root->data->res);
    free(root->data);
    nodes_freed++;
  } else {
    // printf("not freeing a node\n");
    nodes_not_freed++;
  }
  tree_nodes--;

  free_branching_data(root->branching_data);

  // printf("now freed %d nodes, ignored %d\n", nodes_freed, nodes_not_freed);
  return root;
}

double dot(gsl_vector *x, gsl_vector *y) {
  double out;
  gsl_blas_ddot(x, y, &out);
  return out;
}

gsl_vector *obtain_null_vec(gsl_matrix *A) {
  // obtains a vector in the null space of matrix A
  int k = A->size2;
  int d = A->size1;
  gsl_matrix *T = gsl_matrix_alloc(k, k);
  gsl_linalg_QR_decomp_r(A, T);
  // final d-k columns of Q is an ONB for the null space of A^T, assuming A had
  // full rank
  gsl_matrix *Q = gsl_matrix_alloc(d, d);
  gsl_matrix *R = gsl_matrix_alloc(k, k);
  gsl_linalg_QR_unpack_r(A, T, Q, R);

  gsl_vector *v = gsl_vector_alloc(d);
  gsl_matrix_get_col(v, Q, d - 1);

  gsl_matrix_free(T);
  gsl_matrix_free(Q);
  gsl_matrix_free(R);

  return v;
}

gsl_matrix *get_parent_matrix(env_t *env, tnode *node) {
  vsamples_t *vs = env->vsamples;
  size_t d = vs->dimension;

  if (env->params->greer_params.method == 2) {
    // using beam search, so we're already storing this matrix in the node
    gsl_matrix *stored = gsl_matrix_alloc(d, node->depth);
    gsl_matrix_memcpy(
        stored,
        node->parent_basis); // need to copy, since the qr fac may destroy it
    return stored;
  }
  gsl_matrix *A = gsl_matrix_calloc(d, node->depth);
  int k = 0; //= # columns added to A
  tnode *curr = node;
  int expected_cols = node->depth;

  while (curr->depth > 0) {
    // traverse up the tree and grab the sample from every node except the root
    sample_locator_t loc = curr->loc;
    // print_vec(vs->samples[loc.class][loc.index]);
    gsl_matrix_set_col(A, k, vs->samples[loc.class][loc.index]);
    curr = curr->parent;
    k++;
  }
  if (k != expected_cols) {
    printf("[ERR] column error\n");
  }

  return A;
}

gsl_vector *obtain_perp(env_t *env, tnode *node) {
  // obtain a vector orthogonal to all of the samples in node and its ancestors
  // this does a QR factorization - maybe it's faster to do Gram-Schmidt or
  // something else? pretty similar to get_proj_basis - could be refactored
  vsamples_t *vs = env->vsamples;
  size_t d = vs->dimension;
  gsl_matrix *A = get_parent_matrix(env, node);
  if (!A)
    printf("matrix is null, key = %d\n", node->key);
  int k = A->size2;
  /*gsl_matrix *Acopy = gsl_matrix_alloc(d, k); //can be removed
    gsl_matrix_memcpy(Acopy, A);*/

  gsl_vector *v = obtain_null_vec(A);
  double dot_test =
      dot(v, vs->samples[node->loc.class][node->loc.index]); // can be removed
  if (fabs(dot_test) >= 1e-10) {
    printf("[ERR] in obtain_perp: dot = %g\n", dot_test);
    printf("k = %d\n", k);

    /*gsl_vector *b = gsl_vector_alloc(k);
    printf("A = %ld x %ld\n", A->size1, A->size2);
    printf("d = %ld\n", d);
    gsl_blas_dgemv(CblasTrans, 1, A, v, 0, b);
    printf("\nb = ");
    for(int i = 0; i < k; i++)
    printf("%g%s", gsl_vector_get(b, i), i == k - 1 ? "\n" : " ");*/

    exit(1);
  }

  gsl_matrix_free(A);
  return v;
}

tnode *new_empty_child(tnode *node) {
  // creates a new empty child of node and returns a pointer to it
  // children are filled in left->right
  static int curr_key = 1;
  tnode *new = CALLOC(1, tnode);
  new->parent = node;
  new->data = make_hplane_data(NULL, (class_res){0});
  if (!node->l_child) {
    assert(!node->r_child);
    node->l_child = new;
    node->r_child = new;
  } else {
    if (node->r_child->r_sib) {
      printf("[ERR] right child already has a right sibling\n");
    }
    node->r_child->r_sib = new;
    new->l_sib = node->r_child;
    node->r_child = new;
  }
  node->n_children++;
  new->key = curr_key++;
  return new;
}

void add_as_child(tnode *parent, tnode *child) {
  child->parent = parent;
  if (!parent->l_child) {
    assert(!parent->r_child);
    parent->l_child = child;
    parent->r_child = child;
  } else {
    if (parent->r_child->r_sib) {
      printf("[ERR] right child already has a right sibling\n");
    }
    parent->r_child->r_sib = child;
    child->l_sib = parent->r_child;
    parent->r_child = child;
  }
  parent->n_children++;
}

GRBmodel *gr_model;
void create_base_model(env_t *env) {
  int state = 0;
  // double old_theta = env->params->theta;
  // env->params->theta *= 2;
  gr_model = gurobi_relax_within_subspace(&state, env, NULL, 0);
  // env->params->theta = old_theta;
  if (state)
    printf("error creating base gurobi model\n");
}

double *get_rel_hplane(env_t *env, tnode *node) {
  // solves relaxation to get a good hyperplane at the node
  samples_t *samples = env->samples;
  int sz =
      node->depth; // maybe +/-1? should be correct, assuming root has depth 0
  sample_locator_t *basis = CALLOC(sz, sample_locator_t);
  int k = 0;
  while (node->depth > 0) {
    basis[k++] = node->loc;
    node = node->parent;
  }

  assert(k == sz);

  gurobi_add_subspace_constraints(gr_model, env, basis, sz);

  // GRBwrite(gr_model, "tmp_gr.lp");

  int state = GRBoptimize(gr_model);
  if (state) {
    printf("error solving relaxation, state = %d\n", state);
    exit(0);
    return 0;
  }

  int optimstatus;
  state = GRBgetintattr(gr_model, GRB_INT_ATTR_STATUS, &optimstatus);
  if (optimstatus != GRB_OPTIMAL)
    printf("gurobi status is not optimal, status = %d\n", optimstatus);
  // printf("status: %d\n", optimstatus);

  double *h = CALLOC(samples->dimension, double);
  state |=
      GRBgetdblattrarray(gr_model, GRB_DBL_ATTR_X, 0, samples->dimension, h);

  if (state) {
    printf("error getting hplane, state = %d\n", state);
    exit(0);
    return 0;
  }

  free(basis);
  gurobi_remove_last_k_constraints(gr_model, sz);

  /*double *soln = blank_solution(samples);
  hyperplane_to_solution(h, soln, env);
  printf("rel hplane at node %d has prec %g and reach %u\n", node->key,
  precision(soln, samples), reach(soln, samples));*/

  return h;
}

double compute_bound(env_t *env, tnode *node) {
  samples_t *samples = env->samples;

  double *h = get_rel_hplane(env, node);

  double *soln = blank_solution(samples);
  hyperplane_to_solution(h, soln, env);
  double obj = reach(soln, samples);

  free(soln);

  return obj;
}

gsl_vector *arr_to_vec(double *h, int n) {
  gsl_vector *v = gsl_vector_alloc(n);
  gsl_vector view = gsl_vector_view_array(h, n).vector;
  gsl_vector_memcpy(v, &view);

  free(h);

  return v;
}

void copy_parent_basis(env_t *, tnode *);

void create_node_child(env_t *env, tnode *node, sample_locator_t loc,
                       int **branching_data, double score(env_t *, tnode *)) {
  // creates a new child of node with the given loc and computes its vector
  // will create a copy of branching_data
  vsamples_t *vs = env->vsamples;
  tree_nodes++;
  tnode *child = new_empty_child(node);
  child->loc = loc;
  child->depth = node->depth + 1;
  if (DEBUG_GR) {
    printf("Creating child with id %d at depth %d, representing sample (%d, "
           "%ld)\n",
           child->key, child->depth, loc.class, loc.index);
    printf("  Parent id = %d (with sample (%d, %ld))\n", node->key,
           node->loc.class, node->loc.index);
  }
  if (env->params->greer_params.method == 2)
    copy_parent_basis(env, child);
  if (env->params->greer_params.use_rel == 1) {
    child->data->v =
        arr_to_vec(get_rel_hplane(env, child), env->samples->dimension);
  } else {
    child->data->v = obtain_perp(env, child);
  }
  if (DEBUG_GR) {
    printf("  Vector: ");
    print_vec(child->data->v);
  }
  child->branching_data = dup_branching_data(vs, branching_data);

  // need to compute the classification result here in order to compute the
  // score unless we're using cuda, in which case that gets done later
  if (!env->params->greer_params.classify_cuda) {
    child->data->res = classify(env, child->data->v);
    child->score = score(env, child);
    printf("child has score %g\n", child->score);
  }

  if (env->params->greer_params.bnb) {
    double bound = compute_bound(env, child);
    printf("bound = %g, depth = %d\n", bound, child->depth);
    child->score = bound;
  }
}

double zero_score(env_t *env, tnode *node) { return 0; }

double max(double a, double b) { return a > b ? a : b; }

double angle(gsl_vector *a, gsl_vector *b) {
  // returns angle between vectors a and b
  // for now, checks that they are unit vectors
  assert(fabs(gsl_blas_dnrm2(a) - 1) < 1e-10);
  assert(fabs(gsl_blas_dnrm2(b) - 1) < 1e-10);
  return acos(dot(a, b));
}

double dist_penalty(env_t *env, tnode *node) {
  // returns sum of angles from node's hplane to all other known feasible
  // solutions
  double sum = 0;
  for (size_t i = 0; i < env->tree_solution_data.n_solns; i++) {
    sum += fabs(angle(node->data->v, env->tree_solution_data.solutions[i]));
  }
  return sum;
}

double compute_score_penalized(env_t *env, tnode *node) {
  double t = env->params->theta;
  return node->data->res.ntpos - (t / (1 - t)) * node->data->res.nfpos;
}

double compute_score(env_t *env, tnode *node) {
  vsamples_t *vs = env->vsamples;
  if (env->params->greer_params.method == 3) {
    return node->data->res.obj +
           env->params->greer_params.gamma * dist_penalty(env, node);
  }

  return node->data->res.obj;
  // return node->data->res.max_poss_obj;
  // return node->data->res.max_poss_obj*vs->dimension - node->depth;
  // return max(mcts_simulate(env, node).best_max_obj, node->parent->score);
}

int create_node_children_notrim(env_t *env, tnode *node) {
  // creates the children of node. assuming node has a vector and a
  // classification result each child has a vector orthogonal to each sample
  // corresponding to one of its ancestors (except the root, which has no
  // sample) each child will store the resulting vector, along with a locator to
  // the wrongly-classified sample (according to node->v's classification) if
  // the node is already at depth d-1, no children are created

  vsamples_t *vs = env->vsamples;
  if (node->depth >= vs->dimension) {
    printf("attempted to add nodes beyond max depth\n");
    return 0;
  }

  if (node->depth == vs->dimension - 1) {
    if (DEBUG_GR)
      printf("node %d is at max depth, no children added\n", node->key);
    return 0;
  }

  if (DEBUG_GR)
    printf("creating children of node %d\n", node->key);

  int n_added = 0;
  for (size_t i = 0; i < node->data->res.nfneg; i++) {
    // positive samples classified wrong
    sample_locator_t loc =
        (sample_locator_t){.class = 1, .index = node->data->res.fneg[i]};
    create_node_child(env, node, loc, node->branching_data, compute_score);
    n_added++;
  }
  for (size_t i = 0; i < node->data->res.nfpos; i++) {
    // negative samples classified wrong
    sample_locator_t loc =
        (sample_locator_t){.class = 0, .index = node->data->res.fpos[i]};
    create_node_child(env, node, loc, node->branching_data, compute_score);
    n_added++;
  }
  return n_added;
}

int create_node_children_trim(env_t *env, tnode *node) {
  // creates the children of node. assuming node has a vector and a
  // classification result each child has a vector orthogonal to each sample
  // corresponding to one of its ancestors (except the root, which has no
  // sample) each child will store the resulting vector, along with a locator to
  // the wrongly-classified sample (according to node->v's classification) if
  // the node is already at depth d-1, no children are created

  vsamples_t *vs = env->vsamples;
  if (node->depth >= vs->dimension) {
    printf("attempted to add nodes beyond max depth\n");
    return 0;
  }

  if (node->depth == vs->dimension - 1) {
    if (DEBUG_GR)
      printf("node %d is at max depth, no children added\n", node->key);
    return 0;
  }

  if (DEBUG_GR) {
    printf("creating children of node %d\n", node->key);
  }

  int n_added = 0;
  int **curr_b_data = dup_branching_data(vs, node->branching_data);
  for (size_t i = 0; i < node->data->res.nfneg; i++) {
    // positive samples classified wrong
    // printf("considering (%d, %ld), bd[1][%ld] = %d\n", 1, i, i,
    // curr_b_data[1][i]);
    int j = node->data->res.fneg[i];
    if (!curr_b_data[1][j]) {
      curr_b_data[1][j] = 1;
      sample_locator_t loc = (sample_locator_t){.class = 1, .index = j};
      create_node_child(env, node, loc, curr_b_data, compute_score);
      n_added++;
    }
  }
  for (size_t i = 0; i < node->data->res.nfpos; i++) {
    // negative samples classified wrong
    // printf("considering (%d, %ld), bd[0][%ld] = %d\n", 0, i, i,
    // curr_b_data[0][i]);
    int j = node->data->res.fpos[i];
    if (!curr_b_data[0][j]) {
      curr_b_data[0][j] = 1;
      sample_locator_t loc = (sample_locator_t){.class = 0, .index = j};
      create_node_child(env, node, loc, curr_b_data, compute_score);
      n_added++;
    }
  }
  free_branching_data(curr_b_data);

  return n_added;
}

int create_node_children(env_t *env, tnode *node) {
  if (env->params->greer_params.trim == 1)
    return create_node_children_trim(env, node);
  else
    return create_node_children_notrim(env, node);
}

void add_children_to_stack(tnodestack *s, tnode *node) {
  // adds all of node's children to s, in order left -> right
  tnode *curr = node->l_child;
  while (curr) {
    stack_push(s, curr);
    curr = curr->r_sib;
  }
}

int cmp_score(const void *x1, const void *x2) {
  return (*(tnode **)x1)->score - (*(tnode **)x2)->score;
}

int cmp_score_desc(const void *x1, const void *x2) {
  return (*(tnode **)x2)->score - (*(tnode **)x1)->score;
}

void add_children_to_stack_scored(tnodestack *s, tnode *node) {
  // adds all of node's children to s, in decreasing order of their scores
  tnode **children = CALLOC(node->n_children, tnode *);
  double *arr =
      CALLOC(node->n_children,
             double); // TODO - changed this int -> double without testing
  int k = 0;
  tnode *curr = node->l_child;
  while (curr) {
    arr[k] = curr->score;
    children[k++] = curr;
    curr = curr->r_sib;
  }

  qsort(children, node->n_children, sizeof(tnode *), cmp_score);
  for (int i = node->n_children - 1; i >= 0; i--) {
    stack_push(s, children[i]);
  }
  free(children);
}

void add_children_to_stack_scored_trunc(tnodestack *s, tnode *node, int n) {
  // adds the best n of node's children to s, in decreasing order of their
  // scores
  tnode **children = CALLOC(node->n_children, tnode *);
  int k = 0;
  tnode *curr = node->l_child;
  while (curr) {
    children[k++] = curr;
    curr = curr->r_sib;
  }

  qsort(children, node->n_children, sizeof(tnode *), cmp_score);
  int N = n;
  if (N > node->n_children)
    N = node->n_children;
  int i;
  for (i = node->n_children - 1; i >= node->n_children - N; i--) {
    stack_push(s, children[i]);
  }

  // free the children not added
  for (; i >= 0; i--) {
    tree_remove_node(children[i]);
    free(free_subtree(children[i]));
  }
  free(children);
}

void add_children_to_heapq(heapq_t *pq, tnode *node) {
  // adds all of node's children to pq, in order left -> right
  tnode *curr = node->l_child;
  while (curr) {
    tnode *next =
        curr->r_sib; // need to do this now, since enqueue can free curr
    heapq_enqueue(pq, curr);
    curr = next;
  }
}

void flip_res(env_t *env, class_res *res) {
  // replaces res with what it would be if the hyperplane were flipped
  // fneg <-> tpos, fpos <-> tneg, zeros stay the same
  int *temp = res->fneg;
  res->fneg = res->tpos;
  res->tpos = temp;

  int ntemp = res->nfneg;
  res->nfneg = res->ntpos;
  res->ntpos = ntemp;

  temp = res->fpos;
  res->fpos = res->tneg;
  res->tneg = temp;

  ntemp = res->nfpos;
  res->nfpos = res->ntneg;
  res->ntneg = ntemp;

  compute_res_obj(env, res);
}

void check_flip(env_t *env, tnode *node) {
  // checks if a node would be better off with its vector flipped
  // if so, flips it and modifies the classification accordingly
  class_res res = node->data->res;

  // compute what the obj would be after flipping, to decide whether to flip
  hplane_data *n_data = dup_hplane_data(node->data);
  flip_res(env, &n_data->res);
  // int errs = res.nfpos + res.nfneg;
  // int flip_errs = res.ntpos + res.ntneg;
  // if(flip_errs < errs){
  // if(n_data->res.nwrong < res.nwrong) {
  if (n_data->res.obj > res.obj) {
    if (DEBUG_GR)
      printf("flipping node %d\n", node->key);
    // in this case, we'd get a smaller treee by using -v instead of v
    gsl_vector_scale(node->data->v, -1);
    // no need to classify again; the result will be the same, just flipped
    flip_res(env, &node->data->res);
  }
  // TODO: can eliminate the call to flip_res in the if-statement
  free_class_res(&n_data->res);
  gsl_vector_free(n_data->v);
  free(n_data);
}

void list_remove(hplane_list *l, lnode *node) {
  if (node == l->head) {
    l->head = node->next;
    if (l->head)
      l->head->prev = NULL;
  } else if (!node->next) {
    node->prev->next = NULL;
  } else {
    node->prev->next = node->next;
    node->next->prev = node->prev;
  }
  node->data->lists--;
  try_free_data(node->data);
  l->count--;
}

int compute_list_count(hplane_list *l) {
  lnode *curr = l->head;
  int k = 0;
  while (curr) {
    k++;
    curr = curr->next;
  }
  return k;
}

void list_prune(hplane_list *B, int min_obj) {
  // removes everything in list with a lower max possible objective than min_obj
  lnode *curr = B->head;
  lnode *next;
  while (curr) {
    next = curr->next;
    int max_curr_obj = curr->data->res.obj + curr->data->res.npz;
    if (max_curr_obj < min_obj) {
      // remove curr from B
      list_remove(B, curr);
      free(curr);
    }
    curr = next;
  }
  B->poison = 1;
}

void list_compute_range(hplane_list *l) {
  if (!l->poison)
    return;

  lnode *curr = l->head;
  int max_poss = curr->data->res.max_poss_obj,
      min_poss = curr->data->res.obj; // disagrees with greer
  int max_obj = curr->data->res.obj;
  curr = curr->next;
  while (curr) {
    assert(curr->data->lists >= 1);
    max_poss = fmax(max_poss, curr->data->res.max_poss_obj);
    min_poss = fmin(min_poss, curr->data->res.obj);
    max_obj = fmax(max_obj, curr->data->res.obj);
    curr = curr->next;
  }
  l->max_possible_obj = max_poss;
  l->min_possible_obj = min_poss;
  l->max_obj = max_obj;
  l->poison = 0;
}

void remove_list_duplicates(hplane_list *B) {
  // removes all duplicates from B
  // takes O(n^2) time, though it could be done in n log n by sorting B
  lnode *first = B->head;
  while (first) {
    lnode *second = first->next;
    while (second) {
      lnode *next = second->next;
      // if(gsl_vector_equal(first->data->v, second->data->v)) {
      if (dot(first->data->v, second->data->v) > 1 - 1e-8) {
        // printf("dupe: "); print_vec(second->data->v);
        list_remove(B, second);
      }
      second = next;
    }
    first = first->next;
  }
}

void gr_update_b(hplane_list *B, hplane_data *vd) {
  // compares the hyperplane vd to those in B, and updates B to contain all
  // hyperplanes capable of reaching the max known objective

  // TODO: this assertion gets violated
  // assert(vd->lists == 0);

  if (DEBUG_GR) {
    printf("updating B with vector (obj = %g): ", vd->res.obj);
    print_vec(vd->v);
  }

  int max_v_obj = vd->res.max_poss_obj;
  int min_v_obj = vd->res.obj; // disagrees with greer
  list_compute_range(B);
  // this doesn't match exactly what greer says, because his notation doesn't
  // make sense
  if (DEBUG_GR)
    if (max_v_obj > 0)
      printf("max obj = %d\n", max_v_obj);
  if (max_v_obj >= B->min_possible_obj) {
    // in this case, v can potentially be better than anything in B, so add it
    // first, we remove anything that v will always be better than
    list_prune(B, min_v_obj);
    // then add v to B
    list_add_copy(B, vd);
    // then recompute range (since some elements were removed)
    list_compute_range(B);
  }
  try_free_data(vd);
}

void update_b_flips(env_t *env, tnode *curr, hplane_list *B) {
  // considers v and -v (where v = curr->v), and adds them as appropriate to B
  // via update_b
  class_res res_p = curr->data->res, res_n;
  // memcpy(&res_n, &res_p, sizeof(class_res));
  // note - these two are "entangled" - their arrays are the same, so freeing
  // one will free the other
  //  ^ not anymore (hopefully)
  hplane_data *n_data = dup_hplane_data(curr->data);
  flip_res(env, &n_data->res);
  n_data->extra = 1;
  res_n = n_data->res;

  if (res_p.obj > res_n.obj) {
    gr_update_b(B, curr->data);
    try_free_data(n_data);
  } else if (res_p.obj < res_n.obj) {
    gsl_vector_scale(n_data->v, -1);
    gr_update_b(B, n_data);
  } else {
    /*gsl_vector *v_n = gsl_vector_alloc(curr->data->v->size);
    gsl_vector_memcpy(v_n, curr->data->v);
    gsl_vector_scale(v_n, -1);
    hplane_data *n_data = make_hplane_data(v_n, res_n);
    n_data->extra = 1;
    curr->data->res_ent = n_data->res_ent = 1;*/

    gsl_vector_scale(n_data->v, -1);
    gr_update_b(B, curr->data);
    gr_update_b(B, n_data);
  }
}

void gr_update_a(hplane_list *A, hplane_data *vd) {
  if (DEBUG_GR) {
    printf("updating A with vector (obj = %g): ", vd->res.obj);
    print_vec(vd->v);
  }
  list_compute_range(A);
  if (A->count == 0) {
    list_add_copy(A, vd);
  } else {
    if (vd->res.obj >= A->max_obj) {
      if (vd->res.obj > A->max_obj) {
        list_remove_all(A);
      }
      list_add_copy(A, vd);
    }
    try_free_data(vd);
  }
  list_compute_range(A);
}

int dim_zeros(vsamples_t *vs, hplane_data *vd) {
  // computes the dimension of the span of the samples lying on v
  // I'll do this by making them the columns of a matrix, then computing its
  // rank via SVD
  int ncols = vd->res.npz + vd->res.nnz;
  if (ncols == 0)
    return 0;

  size_t d = vs->dimension;
  gsl_matrix *A = gsl_matrix_calloc(d, ncols);
  int k = 0; // current column index
  // positive samples:
  for (int i = 0; i < vd->res.npz; i++) {
    gsl_matrix_set_col(A, k++, vs->samples[1][vd->res.pz[i]]);
  }
  // negative samples
  for (int i = 0; i < vd->res.nnz; i++) {
    gsl_matrix_set_col(A, k++, vs->samples[0][vd->res.nz[i]]);
  }
  if (k != ncols) {
    printf("k != ncols in dim_zeros (%d != %d)\n", k, ncols);
    exit(1); // TODO - remove
  }

  if (ncols > d) {
    // gsl doesn't calculate SVD in this case, so we need to transpose the
    // matrix
    gsl_matrix *AT = gsl_matrix_alloc(A->size2, A->size1);
    gsl_matrix_transpose_memcpy(AT, A);
    gsl_matrix_free(A);
    A = AT;
    ncols = d;
    // nrows = old ncols, but that isn't needed
  }
  gsl_vector *s = gsl_vector_alloc(ncols);
  gsl_matrix *V = gsl_matrix_alloc(ncols, ncols);
  gsl_vector *work = gsl_vector_alloc(ncols);

  gsl_linalg_SV_decomp(A, V, s, work);

  int dim = 0;
  for (int i = 0; i < ncols; i++) {
    double si = gsl_vector_get(s, i);
    if (fabs(si) > 1e-10)
      dim++;
  }
  assert(dim <= d);

  gsl_vector_free(s);
  gsl_matrix_free(V);
  gsl_matrix_free(A);
  gsl_vector_free(work);
  return dim;
}

double comp_disp(vsamples_t *vs, gsl_vector *x, gsl_vector *z) {
  double min = 1e101;
  for (int class = 0; class < vs->class_cnt; class ++) {
    for (int i = 0; i < vs->count[class]; i++) {
      // positive samples
      double x_dot_y, z_dot_y;
      gsl_blas_ddot(x, vs->samples[class][i], &x_dot_y);
      if (fabs(x_dot_y) < 1e-10) // then y is on the hyperplane, so ignore it
        continue;

      gsl_blas_ddot(z, vs->samples[class][i], &z_dot_y);

      if ((x_dot_y > 0 && z_dot_y < 0) || (x_dot_y < 0 && z_dot_y > 0)) {
        min = fmin(min, x_dot_y / (x_dot_y - z_dot_y));
      }
    }
  }
  if (min == 1e101)
    return 0.5;
  return min / 2;
}

void update_a_displace(env_t *env, hplane_data *vd, gsl_vector *y,
                       hplane_list *A) {
  // helper function for solve_dim1 and solve_pc, where we have to displace vd.v
  // towards y, then add it to A
  vsamples_t *vs = env->vsamples;
  double alpha = comp_disp(vs, vd->v, y);
  gsl_vector *displaced = gsl_vector_alloc(y->size);
  gsl_vector_memcpy(displaced, vd->v);
  if (DEBUG_GR) {
    printf("alpha = %g\n", alpha);
    printf("v = ");
    print_vec(vd->v);
    printf("y = ");
    print_vec(y);
  }
  gsl_vector_axpby(alpha, y, 1 - alpha, displaced); // v <- alpha y + (1-alpha)v

  // TODO: possibly scale the displaced vector, since it won't have unit norm
  // anymore
  hplane_data *displaced_data =
      make_hplane_data(displaced, classify(env, displaced));
  displaced_data->extra = 0;

  // printf("update_a_displace info:\n#zeros = %d\nv = ", vd->res.npz +
  // vd->res.nnz); print_vec(vd->v); printf("y = "); print_vec(y); printf("v obj
  // = %g\n", vd->res.obj); printf("y obj = %g\n", classify(env, y).obj);
  // printf("resulting obj = %g\n", displaced_data->res.obj);
  gr_update_a(A, displaced_data);
  gsl_vector_free(displaced);
  free_class_res(&displaced_data->res);
  free(displaced_data);
}

void solve_dim1(env_t *env, hplane_data *vd, hplane_list *A) {
  vsamples_t *vs = env->vsamples;
  gsl_vector *y;
  // y needs to be assigned to something on the hplane
  // preferring positive samples here, but I don't know if that's best
  assert(vd->res.npz + vd->res.nnz > 0);
  // if y is a negative sample, need to copy it and flip it
  int class_label;
  if (vd->res.npz > 0) {
    y = vs->samples[1][vd->res.pz[0]];
    class_label = 1;
  } else {
    y = gsl_vector_alloc(vd->v->size);
    gsl_vector_memcpy(y, vs->samples[0][vd->res.nz[0]]);
    gsl_vector_scale(y, -1);
    class_label = -1;
  }

  update_a_displace(env, vd, y, A);

  if (class_label == -1)
    gsl_vector_free(y);

  // TODO (important): if there is a sample equal to -y, need to do extra
  // but that won't likely come up in real datasets
  // see greer book pg 104 bottom for how to deal with it
}

void solve_pc(env_t *env, hplane_data *vd, gsl_vector *x, hplane_list *A) {
  update_a_displace(env, vd, x, A);
}

vsamples_t *construct_subvs(vsamples_t *vs, hplane_data *vd) {
  // returns a new vsamples with only those samples that lie on the hplane vd
  // note that it doesn't copy the sample vectors, so it should be freed using
  // the below function
  vsamples_t *nvs = malloc(sizeof(vsamples_t));
  memcpy(nvs, vs, sizeof(vsamples_t));
  nvs->count = malloc(nvs->class_cnt * sizeof(size_t));
  nvs->count[0] = vd->res.nnz;
  nvs->count[1] = vd->res.npz;
  nvs->samples = CALLOC(nvs->class_cnt, gsl_vector **);

  nvs->samples[0] = CALLOC(nvs->count[0], gsl_vector *);
  for (int i = 0; i < nvs->count[0]; i++) {
    nvs->samples[0][i] = vs->samples[0][i];
  }

  nvs->samples[1] = CALLOC(nvs->count[1], gsl_vector *);
  for (int i = 0; i < nvs->count[1]; i++) {
    nvs->samples[1][i] = vs->samples[1][i];
  }
  return nvs;
}

void free_subvs(vsamples_t *nvs) {
  free(nvs->samples[0]);
  free(nvs->samples[1]);
  free(nvs->samples);
  free(nvs->count);
  free(nvs);
}

gsl_vector *max_accuracy_random_hyperplane(env_t *, int);

void solve_npc(env_t *env, hplane_data *vd, hplane_list *A, int dim) {
  // need a recursive call in this case
  printf("recursive call needed\n");
  printf("npz = %d, nnz = %d, dim = %d\n", vd->res.npz, vd->res.nnz, dim);

  vsamples_t *vs = env->vsamples;

  vsamples_t *nvs = construct_subvs(vs, vd);
  env->vsamples =
      nvs; // TODO: change this to make a new env for the recursive call

  /*gsl_vector *v0 = gsl_vector_alloc(vd->v->size);
    gsl_vector_memcpy(v0, vd->v);*/
  gsl_vector *v0 = max_accuracy_random_hyperplane(env, 10000);

  env->vsamples =
      nvs; // TODO: change this to make a new env for the recursive call
  hplane_list *nA = gr_explore(env, v0, create_list());
  env->vsamples = vs;

  lnode *curr = nA->head;
  while (curr) {
    update_a_displace(env, vd, curr->data->v, A);
    curr = curr->next;
  }
  free(list_free(nA));
  free_subvs(nvs);
}

void solve_highdim(env_t *env, hplane_data *vd, hplane_list *A, int dim) {
  if (DEBUG_GR) {
    printf("solving LP, starting with the vector (obj = %g): ", vd->res.obj);
    print_vec(vd->v);
  }

  vsamples_t *vs = env->vsamples;
  pc_soln sol = detect_pointed_cone(vs, *vd);
  double gamma = sol.gamma;
  gsl_vector *x = sol.x;
  if (DEBUG_GR) {
    printf("result: gamma = %g, x = ", gamma);
    print_vec(x);
  }
  assert(gamma == 0 || gamma == 1);

  switch ((int)gamma) {
  case 1:
    solve_pc(env, vd, x, A);
    break;
  case 0:
    if (env->params->greer_params.skip_rec)
      break;
    solve_npc(env, vd, A, dim);
    return;
    break;
  default:
    printf("[ERR] gamma = %g\n", gamma);
    exit(1);
  }

  gsl_vector_free(x);
}

void displace_single(env_t *env, hplane_data *vd, hplane_list *A) {
  int sub_dim = dim_zeros(env->vsamples, vd);
  if (DEBUG_GR)
    printf("sub_dim = %d\n", sub_dim);

  vd->extra = 1; // allow update_a to free curr
  switch (sub_dim) {
  case 0:
    gr_update_a(A, vd);
    break;
  case 1:
    solve_dim1(env, vd, A);
    try_free_data(vd);
    break;
  default:
    solve_highdim(env, vd, A, sub_dim);
    try_free_data(vd);
  }
}

hplane_list *gr_displace(env_t *env, hplane_list *B, hplane_list *A) {
  if (DEBUG_GR) {
    printf("entered displace. B = \n");
    print_hplane_list(B);
  }

  vsamples_t *vs = env->vsamples;
  remove_list_duplicates(B);
  printf("%d unique boundary vectors\n", B->count);

  lnode *curr = B->head, *next;
  while (curr) {
    if (env->params->greer_params.no_displace) {
      list_add(A, curr->data);
    } else {
      displace_single(env, curr->data, A);
    }

    next = curr->next;
    free(curr);
    curr = next;
  }
  return A;
}

void replace_with_copies(hplane_list *l) {
  // replaces every non-extra vector in l with a copy, so that the tree can be
  // freed also copies the classification results for the same reason
  lnode *curr = l->head;
  while (curr) {
    if (!curr->data->extra) {
      gsl_vector *v = curr->data->v;
      gsl_vector *cp = gsl_vector_alloc(v->size);
      gsl_vector_memcpy(cp, v);
      curr->data->v = cp;
      curr->data->extra = 1;
    }
    curr = curr->next;
  }
}

void set_all_extra(hplane_list *l) {
  // set all entries in l to be "extra"
  lnode *curr = l->head;
  while (curr) {
    curr->data->extra = 1;
    curr = curr->next;
  }
}

int subtree_explored(tnode *node) {
  // checks if an entire subtree has been explored
  // note: this is very slow
  if (node->explored == 0) {
    printf("node %d not explored", node->key);
    if (node->parent) {
      printf(", with parent %d\n", node->parent->key);
    } else
      printf("\n");

    return 0;
  }
  tnode *curr = node->l_child;
  if (!curr) {
    assert(!node->r_child);
  }
  int out = 1;
  while (curr) {
    out &= subtree_explored(curr);
    curr = curr->r_sib;
  }
  return out;
}

int print_children(tnode *node) {
  int c = 0;
  tnode *curr = node->l_child;
  printf("  children: ");
  while (curr) {
    printf("%d ", curr->key);
    c++;
    curr = curr->r_sib;
  }
  printf("\n");
  return c;
}

void check_free(tnode *node, int trunc) {
  // given a max-depth node, frees as much of its subtree as possible
  // to do this, it sets the variables children_explored in some of its
  // ancestors

  if (node->parent)
    node->parent->children_explored++;

  tnode *curr = node;
  // int max_children = trunc == 0 ? curr->parent->n_children : trunc;
  // printf("entered check_free, node %d, trunc = %d\n", node->key, trunc);
  // printf("max_children = %d\n", max_children);
  while (curr->parent &&
         (curr->parent->children_explored >=
          ((trunc == 0) ? curr->parent->n_children : trunc)) &&
         curr->parent->depth > 0) {
    curr = curr->parent;
    // printf("moved back to node %d, which has %d children explored of %d
    // total\n", curr->key, curr->children_explored, curr->n_children); printf("
    // counted %d children\n", print_children(curr)); at this point, we know
    // that curr has been fully explored so we can update its parent
    curr->parent->children_explored++;
  }

  if (DEBUG_GR)
    printf("about to free subtree starting at node %d, depth %d\n", curr->key,
           curr->depth);
  if (!subtree_explored(curr)) { // this is slow, only for testing
    // and not valid for truncated trees
    printf("subtree not fully explored\n");
    printf("path: ");
    tnode *curr2 = node;
    while (curr2 != curr) {
      printf("%d ", curr2->key);
      curr2 = curr2->parent;
    }
    printf("%d\n", curr2->key);
    exit(0);
  }

  if (curr->in_q)
    printf("check_free - node in q\n");
  tree_remove_node(curr);
  free(free_subtree(curr));
}

hplane_list *gr_explore(env_t *env, gsl_vector *v0, hplane_list *A) {
  vsamples_t *vs = env->vsamples;
  size_t d = vs->dimension;
  class_res root_res = classify(env, v0);
  tnode root = (tnode){make_hplane_data(v0, root_res),
                       .branching_data = blank_branching_data(vs),
                       .key = 0}; // set vector v0 and depth = 0, nothing else
  if (DEBUG_GR) {
    printf("v0 = ");
    print_vec(v0);
  }

  // params:
  int use_heapq = env->params->greer_params.use_heapq;
  int trunc = env->params->greer_params.trunc;
  size_t (*get_size)(void *);
  tnode *(*pop)(void *);
  void *coll;
  if (use_heapq) {
    heapq_t *pq = create_heapq(env->params->greer_params.max_heapq_size);
    heapq_enqueue(pq, &root);
    get_size = (size_t (*)(void *))heapq_get_n;
    pop = (tnode * (*)(void *)) heapq_pop_max;
    coll = pq;
  } else {
    tnodestack *s = create_stack();
    stack_push(s, &root);
    get_size = (size_t (*)(void *))stack_get_n;
    pop = (tnode * (*)(void *)) stack_pop;
    coll = s;
  }

  double best_obj = -1e101; // temporary - no need to fill in now
  gsl_vector *best_v = v0;
  hplane_list *B = create_list();
  list_add_copy(B, root.data);

  static int nodes_explored = 0; // needs to be static, since we restart

  time_t start = time(0);

  int k = 0;
  // while(!stack_empty(s)) {
  // while(heapq_get_n(pq) > 0) {
  while (get_size(coll) > 0) {
    /*if(nodes_explored > 100)
      break;*/
    // tnode *curr = stack_pop(s);
    // tnode *curr = heapq_pop_max(pq);
    // tnode *curr = heapq_pop_epsilon(pq, 0.01);
    tnode *curr = pop(coll);
    if (DEBUG_GR)
      printf("popped from stack, id = %d, score = %g\n", curr->key,
             curr->score);

    /*double dupe_arr[] = {-0.164399, -5.55112e-17, 2.99101e-14, -9.24781e-15,
    -1.93075e-15, -2.72005e-15, 0.986394}; gsl_vector dupe =
    gsl_vector_view_array(dupe_arr, d).vector; if(dot(curr->data->v, &dupe) >
    1-1e-8) { printf("found duplicate, path = "); tnode *trav = curr;
      while(trav) {
        printf("%d%s", trav->key, trav->parent == NULL ? "\n" : " <- ");
        trav = trav->parent;
      }
      }*/

    nodes_explored++;
    if (time(0) - start > 0) {
      // printf("%d nodes explored, %d items in stack, %d nodes in tree, best
      // obj = %g\n", nodes_explored, s->top, tree_nodes, best_obj); printf("%d
      // nodes explored, %ld items in pq, %d nodes in tree, best obj = %g\n",
      // nodes_explored, pq->n, tree_nodes, best_obj);
      printf("%d nodes explored, %ld items in queue, %d nodes in tree, best "
             "obj = %g\n",
             nodes_explored, get_size(coll), tree_nodes, best_obj);
      start = time(0);
    }

    // find the vector v for curr if it doesn't already exist
    if (!curr->data->v) { // must be the root node in this case
      // actually, this is unncessary, since the root node will already have v0
      printf("[ERR] found node without a vector\n");
      curr->data->v = v0;
    }
    // classify by v and store it in the node
    /*class_res res = classify(vs, curr->data->v);
      curr->data->res = res;*/
    // it is already getting classified when it is created, so no need
    class_res res = curr->data->res;
    check_flip(env, curr);

    if (curr->data->res.obj > best_obj) {
      best_obj = curr->data->res.obj;
      best_v = curr->data->v;
      printf("new best soln with objective %g\n", best_obj);
      print_vec(best_v);
    }

    if (res.nwrong == 0) {
      printf("perfect vector found at node %d\n", curr->key);
      list_remove_all(B);
      list_add_copy(B, curr->data);
      gr_displace(env, B, A);
      return A;
    }

    /*if(res.nfneg + res.nfpos + res.nnz + res.npz - curr->depth <
      root.data->res.nfneg + root.data->res.nfpos && !trunc) {
      //in this case, we're better off restarting with the current vector
      //since it will lead to a smaller tree
      list_remove_all(B);
      list_add_copy(B, curr->data);

      gr_displace(vs, B, A);
      gsl_vector *new_v0 = gsl_vector_alloc(v0->size);
      gsl_vector_memcpy(new_v0, A->head->data->v);

      free(B); //displace will already have freed everything in B
      free(stack_free(s));

      free_subtree(&root);
      printf("restarting, new solution has at most %d errors (A has %d
      hplane(s))\n", res.nfneg + res.nfpos + res.nnz + res.npz - curr->depth,
      A->count); return gr_explore(vs, new_v0, A, trunc);
      }*/

    create_node_children(env, curr);
    if (curr->depth < d - 1) {
      // add_children_to_stack(s, curr);
      if (use_heapq) {
        add_children_to_heapq((heapq_t *)coll, curr);
      } else {
        if (trunc)
          add_children_to_stack_scored_trunc((tnodestack *)coll, curr, trunc);
        else
          add_children_to_stack_scored((tnodestack *)coll, curr);
      }
      // add_children_to_heapq(pq, curr);
      if (DEBUG_GR)
        printf("calling update_b\n");
      if (curr->key != 0)
        gr_update_b(B, curr->data); // no need to add the root twice
    } else {
      if (DEBUG_GR)
        printf("calling update_b_flips\n");
      update_b_flips(env, curr, B);
    }

    // if possible, we want to free the subtree that we've already explored
    /*if(curr->parent) {
      curr->parent->children_explored++;
      }*/
    assert(curr->explored == 0);
    curr->explored = 1;
    if (curr->depth == d - 1) {
      check_free(curr, trunc);
    } else {
      try_free_class_res(&curr->data->res);
    }
  }
  printf("loop done\n");
  // free the tree
  // before that, we need to make sure all of the vectors in B have been copied,
  // so that we don't free them
  /*printf("replacing B with copies\n");
    replace_with_copies(B);*/

  // now we can free it
  free_subtree(&root);

  if (use_heapq)
    free(heapq_free(coll));
  else
    free(stack_free(coll));

  // exit(0);

  printf("finished tree search, now displacing %d boundary vectors\n",
         B->count);

  gr_displace(env, B, A);
  printf("best obj = %d\n", A->max_obj);
  printf("explored %d nodes\n", nodes_explored);

  free(B);

  return A;
}

gsl_vector *max_accuracy_random_hyperplane(env_t *env, int N) {
  // returns the best of N random hyperplanes, where "best" means the one which
  // has the fewest misclassifications
  vsamples_t *vs = env->vsamples;
  size_t d = vs->dimension;

  gsl_vector *v = gsl_vector_calloc(d);
  int best_n_errs = vs->count[0] + vs->count[1];
  for (int i = 0; i < N; i++) {
    double *h = CALLOC(d, double);
    random_unit_vector(d, h);
    gsl_vector curr = gsl_vector_view_array(h, d).vector;
    class_res res = classify(env, &curr);
    int n_errs = res.nfpos + res.nfneg;
    if (n_errs < best_n_errs) {
      best_n_errs = n_errs;
      gsl_vector_memcpy(v, &curr);
    }
    free(h);
    free_class_res(&res);
  }
  printf("best hyperplane has %d errors\n", best_n_errs);
  return v;
}

gsl_vector *successive_trunc(env_t *env, gsl_vector *v0, int start, int inc,
                             int max) {
  printf("Running successive truncation\n");

  vsamples_t *vs = env->vsamples;
  size_t d = vs->dimension;
  for (int trunc = start; trunc <= max; trunc += inc) {
    printf("Starting trunc = %d\n", trunc);
    env->params->greer_params.trunc = trunc;
    hplane_list *A = gr_explore(env, v0, create_list());
    v0 = gsl_vector_alloc(d);
    // printf("v0 size = %ld, A head size = %ld\n", v0->size,
    // A->head->data->v->size);
    gsl_vector_memcpy(v0, A->head->data->v);
    free(list_free(A));
    printf("finished trunc = %d\n  v = ", trunc);
    print_vec(v0);
    printf("  obj = %g\n", classify(env, v0).obj);
  }
  return v0;
}

void print_beam(tnode **beam, int width) {
  for (int i = 0; i < width; i++) {
    if (beam[i])
      printf("%g ", beam[i]->score);
    else
      printf("/ ");
  }
  printf("\n");
}

int add_children_to_arr(tnode **arr, tnode *node, int start) {
  tnode *curr = node->l_child;
  while (curr) {
    arr[start++] = curr;
    curr = curr->r_sib;
  }
  return start;
}

int count_children(tnode *node) {
  int n = 0;
  tnode *curr = node->l_child;
  while (curr) {
    curr = curr->r_sib;
    n++;
  }
  return n;
}

void copy_parent_basis(env_t *env, tnode *node) {
  // alloc's node->parent_samples, copies it from parent, and copies parent's
  // sample into it
  if (node->depth == 0)
    return;
  if (node->parent_basis)
    return; // if basis already exists, no need to make a new one
  vsamples_t *vs = env->vsamples;
  size_t d = vs->dimension;
  int depth = node->depth;
  node->parent_basis = gsl_matrix_calloc(d, depth);
  // copy first depth-1 cols from parent:
  for (int i = 0; i < depth - 1; i++) {
    gsl_vector col =
        gsl_matrix_const_column(node->parent->parent_basis, i).vector;
    gsl_matrix_set_col(node->parent_basis, i + 1, &col);
  }
  sample_locator_t loc = node->loc;
  gsl_matrix_set_col(node->parent_basis, 0, vs->samples[loc.class][loc.index]);
}

tnode *free_node_alone(tnode *node) {
  // frees everything stored in the node, and does nothing else
  gsl_vector_free(node->data->v);
  try_free_class_res(&node->data->res);
  free(node->data);
  free_branching_data(node->branching_data);
  if (node->parent_basis)
    gsl_matrix_free(node->parent_basis);
  return node;
}

void score_all_from_mat(env_t *env, gsl_matrix *mat, tnode **children,
                        int n_children) {
  // mat is matrix SW of dot product results
  // this is O(mn), so it'd be great to do on GPU, but difficult to work with
  // the data strctures
  //   otherwise it seems quite easy to parallelize
  vsamples_t *vs = env->vsamples;
  size_t n = samples_total(env->samples);
  size_t d = vs->dimension;

  for (int j = 0; j < mat->size2; j++) {
    if (!children[j]) {
      continue;
    }
    for (int i = 0; i < mat->size1; i++) {
      double elt = gsl_matrix_get(mat, i, j);
      if (i < vs->count[0]) {
        // negative samples
        if (fabs(elt) < 1e-10) {
          children[j]->data->res.nnz++;
        } else if (elt > 0) {
          children[j]->data->res.nfpos++;
        } else {
          children[j]->data->res.ntneg++;
        }
      } else {
        if (fabs(elt) < 1e-10) {
          children[j]->data->res.nnz++;
        } else if (elt > 0) {
          children[j]->data->res.ntpos++;
        } else {
          children[j]->data->res.ntneg++;
        }
      }
    }
    children[j]->data->res.nwrong =
        children[j]->data->res.nfneg + children[j]->data->res.nfpos;
    children[j]->data->res.nright =
        children[j]->data->res.ntpos + children[j]->data->res.ntneg;

    compute_res_obj(env, &children[j]->data->res);
    children[j]->score = compute_score(env, children[j]);
    // printf("computed score %g for node %d\n",     children[j]->score,
    // children[j]->key);
  }
}

int beam_iter(env_t *env, tnode **beam, size_t width) {
  int n_children = 0;
  int n_in_beam = 0;
  for (int i = 0; i < width; i++) {
    if (beam[i] != NULL) {
      create_node_children(env, beam[i]);
      int nc = beam[i]->n_children;
      n_children += nc;
      n_in_beam++;
    }
  }

  tnode **children = CALLOC(n_children, tnode *);
  int k = 0;
  for (int i = 0; i < width; i++) {
    if (!beam[i])
      continue;
    k = add_children_to_arr(children, beam[i], k);
    // children[k++] = beam[i];
  }

  /*if(env->params->greer_params.classify_cuda) {
    printf("generated %d children\n", n_children);
    int batch_size = n_children;
    for(int i = 0; i < n_children-batch_size; i += batch_size) {
      gsl_matrix *mat = classify_many(env, children+i, fmin(n_children - i,
  batch_size)); printf("matrix has shape (%ld, %ld)\n", mat->size1, mat->size2);
      score_all_from_mat(env, mat, children+i, fmin(n_children-i, batch_size));
      gsl_matrix_free(mat);
    }
  }*/

  qsort(children, k, sizeof(tnode *), cmp_score_desc);

  // free old beam:
  for (int i = 0; i < width; i++) {
    if (!beam[i])
      continue;
    tnode *node = beam[i];
    free_node_alone(node);
    if (node->depth > 0)
      free(node);
  }

  for (int i = 0; i < width; i++) {
    if (i >= k)
      beam[i] = NULL;
    else {
      // tree_remove_node(children[i]);
      beam[i] = children[i];
    }
  }
  for (int i = width; i < n_children; i++) {
    if (children[i]->depth > 0) {
      tnode *node = children[i];
      free(free_node_alone(node));
    }
  }

  free(children);

  return (n_children == 0) ? 0 : 1;
}

int create_node_children_pq(env_t *env, tnode *node, heapq_t *pq) {
  // creates the children of node and inserts them into capped priority queue

  vsamples_t *vs = env->vsamples;
  if (node->depth >= vs->dimension) {
    printf("attempted to add nodes beyond max depth\n");
    return 0;
  }

  if (node->depth == vs->dimension - 1) {
    if (DEBUG_GR)
      printf("node %d is at max depth, no children added\n", node->key);
    return 0;
  }

  if (DEBUG_GR)
    printf("creating children of node %d\n", node->key);

  int n_added = 0;
  for (size_t i = 0; i < node->data->res.nfneg; i++) {
    // positive samples classified wrong
    sample_locator_t loc =
        (sample_locator_t){.class = 1, .index = node->data->res.fneg[i]};
    create_node_child(env, node, loc, node->branching_data, compute_score);
    heapq_enqueue(pq, node->r_child);
    n_added++;
  }
  for (size_t i = 0; i < node->data->res.nfpos; i++) {
    // negative samples classified wrong
    sample_locator_t loc =
        (sample_locator_t){.class = 0, .index = node->data->res.fpos[i]};
    create_node_child(env, node, loc, node->branching_data, compute_score);
    heapq_enqueue(pq, node->r_child);
    n_added++;
  }
  return n_added;
}

int beam_iter_pq(env_t *env, tnode **beam, size_t width) {
  int n_children = 0;
  int n_in_beam = 0;
  heapq_t *pq = create_heapq(width);

  for (int i = 0; i < width; i++) {
    if (beam[i] != NULL) {
      create_node_children_pq(env, beam[i], pq);
      int nc = beam[i]->n_children;
      n_children += nc;
      n_in_beam++;
    }
  }

  /*tnode **children = CALLOC(n_children, tnode *);
  int k = 0;
  for(int i = 0; i < width; i++){
    if(!beam[i]) continue;
    k = add_children_to_arr(children, beam[i], k);
    //children[k++] = beam[i];
  }

  if(env->params->greer_params.classify_cuda) {
    printf("generated %d children\n", n_children);
    int batch_size = n_children;
    for(int i = 0; i < n_children-batch_size; i += batch_size) {
      gsl_matrix *mat = classify_many(env, children+i, fmin(n_children - i,
  batch_size)); printf("matrix has shape (%ld, %ld)\n", mat->size1, mat->size2);
      score_all_from_mat(env, mat, children+i, fmin(n_children-i, batch_size));
      gsl_matrix_free(mat);
    }
    }


    qsort(children, k, sizeof(tnode *), cmp_score_desc);

  //free old beam:
  for(int i = 0; i < width; i++) {
    if(!beam[i]) continue;
    tnode *node = beam[i];
    free_node_alone(node);
    if(node->depth > 0)
      free(node);
  }

  for(int i = 0; i < width; i++) {
    if(i >= k)
      beam[i] = NULL;
    else {
      //tree_remove_node(children[i]);
      beam[i] = children[i];
    }
  }
  for(int i = width; i < n_children; i++) {
    if(children[i]->depth > 0) {
      tnode *node = children[i];
      free(free_node_alone(node));
    }
  }

  free(children);*/
  if (n_children >= width)
    memcpy(beam, pq->heap, width * sizeof(tnode *));
  else {
    memcpy(beam, pq->heap, width * sizeof(tnode *));
    memset(beam + n_children, 0, (width - n_children) * sizeof(tnode *));
  }

  return (n_children == 0) ? 0 : 1;
}

void init_basis(env_t *env, tnode *root) {
  vsamples_t *vs = env->vsamples;
  size_t d = vs->dimension;
  root->parent_basis = gsl_matrix_alloc(d, 0);
}

hplane_list *beam_search(env_t *env, gsl_vector *v0, hplane_list *A) {
  vsamples_t *vs = env->vsamples;
  size_t d = vs->dimension;
  class_res root_res = classify(env, v0);
  tnode root = (tnode){make_hplane_data(v0, root_res),
                       .branching_data = blank_branching_data(vs),
                       .key = 0}; // set vector v0 and depth = 0, nothing else
  if (DEBUG_GR) {
    printf("v0 = ");
    print_vec(v0);
  }

  int width = env->params->greer_params.beam_width;
  tnode **beam = CALLOC(width, tnode *);
  beam[0] = &root;

  root.score = compute_score(env, &root);
  init_basis(env, &root);

  double best_obj = root.data->res.obj;
  gsl_vector *best_v = root.data->v;
  hplane_list *B = create_list();
  list_add_copy(B, root.data);

  static int nodes_explored = 0; // needs to be static, since we restart

  time_t start = time(0);

  int k = 0;
  int done = 0;
  while (!done) {
    if (!beam[0])
      break; // TODO - why this happens
    /*if(time(0) - start >= 120) {
      printf("time\n");
      break;
      }*/
    printf("depth = %d\n", beam[0]->depth);
    /*printf("beam:\n");
      print_beam(beam, width);*/
    beam_iter(env, beam, width);
    // beam_iter_pq(env, beam, width);

    for (int i = 0; i < width; i++) {
      if (!beam[i])
        continue;
      tnode *curr = beam[i];
      if (env->params->greer_params.classify_cuda)
        curr->data->res = classify(
            env, curr->data->v); // this repeats computation, but it is needed
      class_res res = curr->data->res;
      check_flip(env, curr);

      if (curr->data->res.obj > best_obj) {
        best_obj = curr->data->res.obj;
        best_v = curr->data->v;
        printf("new best soln with objective %g\n", best_obj);
      }

      if (res.nwrong == 0) {
        printf("perfect vector found at node %d\n", curr->key);
        list_remove_all(B);
        list_add_copy(B, curr->data);
        gr_displace(env, B, A);
        return A;
      }

      if (curr->depth < d - 1) {
        if (curr->key != 0)
          gr_update_b(B, curr->data); // no need to add the root twice
      } else {
        if (DEBUG_GR)
          printf("calling update_b_flips\n");
        update_b_flips(env, curr, B);
      }
      assert(curr->explored == 0);
      curr->explored = 1;
      /*if(curr->depth == d - 1) {
        check_free(curr, 0);
      } else {
        try_free_class_res(&curr->data->res);
        }*/
    }
  }
  printf("loop done\n");
  // free the tree
  // before that, we need to make sure all of the vectors in B have been copied,
  // so that we don't free them
  /*printf("replacing B with copies\n");
    replace_with_copies(B);*/

  // now we can free it
  //   free_subtree(&root);
  // exit(0);

  printf("finished tree search, now displacing %d boundary vectors\n",
         B->count);

  gr_displace(env, B, A);
  printf("best obj = %d\n", A->max_obj);
  printf("explored %d nodes\n", nodes_explored);

  free(B);

  return A;
}

hplane_list *prec_beam_search(env_t *env, gsl_vector *v0, hplane_list *A) {
  vsamples_t *vs = env->vsamples;
  size_t d = vs->dimension;
  class_res root_res = classify(env, v0);
  tnode root = (tnode){make_hplane_data(v0, root_res),
                       .branching_data = blank_branching_data(vs),
                       .key = 0}; // set vector v0 and depth = 0, nothing else
  if (DEBUG_GR) {
    printf("v0 = ");
    print_vec(v0);
  }

  int width = env->params->greer_params.beam_width;
  tnode **beam = CALLOC(width, tnode *);
  beam[0] = &root;

  root.score = compute_score(env, &root);
  init_basis(env, &root);

  double best_obj = root.data->res.obj;
  gsl_vector *best_v = root.data->v;
  hplane_list *B = create_list();
  list_add_copy(B, root.data);

  static int nodes_explored = 0; // needs to be static, since we restart

  time_t start = time(0);

  int k = 0;
  int done = 0;
  while (!done) {
    if (!beam[0])
      break; // TODO - why this happens
    printf("depth = %d\n", beam[0]->depth);
    /*printf("beam:\n");
      print_beam(beam, width);*/
    beam_iter(env, beam, width);
    // beam_iter_pq(env, beam, width);

    for (int i = 0; i < width; i++) {
      if (!beam[i])
        continue;
      tnode *curr = beam[i];
      if (env->params->greer_params.classify_cuda)
        curr->data->res = classify(
            env, curr->data->v); // this repeats computation, but it is needed
      class_res res = curr->data->res;
      check_flip(env, curr);

      if (curr->data->res.obj > best_obj) {
        best_obj = curr->data->res.obj;
        best_v = curr->data->v;
        printf("new best soln with objective %g\n", best_obj);
      }

      if (res.nwrong == 0) {
        printf("perfect vector found at node %d\n", curr->key);
        list_remove_all(B);
        list_add_copy(B, curr->data);
        gr_displace(env, B, A);
        return A;
      }

      if (curr->depth < d - 1) {
        if (curr->key != 0)
          gr_update_b(B, curr->data); // no need to add the root twice
      } else {
        if (DEBUG_GR)
          printf("calling update_b_flips\n");
        update_b_flips(env, curr, B);
      }
      assert(curr->explored == 0);
      curr->explored = 1;
      /*if(curr->depth == d - 1) {
        check_free(curr, 0);
      } else {
        try_free_class_res(&curr->data->res);
        }*/
    }
  }
  printf("loop done\n");
  // free the tree
  // before that, we need to make sure all of the vectors in B have been copied,
  // so that we don't free them
  /*printf("replacing B with copies\n");
    replace_with_copies(B);*/

  // now we can free it
  //   free_subtree(&root);
  // exit(0);

  printf("finished tree search, now displacing %d boundary vectors\n",
         B->count);

  gr_displace(env, B, A);
  printf("best obj = %d\n", A->max_obj);
  printf("explored %d nodes\n", nodes_explored);

  free(B);

  return A;
}

double *single_greer_run(env_t *env, double *h0) {
  vsamples_t *vs = samples_to_vec(env->samples);
  // store_samples_cuda(env->samples);
  // envparams = env->params;
  env->vsamples = vs;
  size_t d = env->samples->dimension;

  /*if (env->params->greer_params.classify_cuda)
    store_samples_cuda(env->samples);*/

  if (env->params->greer_params.bnb == 1 ||
      env->params->greer_params.use_rel == 1)
    create_base_model(env);

  gsl_vector *v0;
  if (h0) {
    gsl_vector hvec = gsl_vector_view_array(h0, d).vector;
    v0 = gsl_vector_calloc(vs->dimension);
    gsl_vector_memcpy(v0, &hvec);
  } else {
    v0 = max_accuracy_random_hyperplane(env, 10000);
    env->params->greer_params.obj_code = WRC;
    gsl_vector_scale(v0, 1 / gsl_blas_dnrm2(v0));
    env->params->greer_params.trunc = 10;
    hplane_list *A = gr_explore(env, v0, create_list());
    v0 = gsl_vector_alloc(d);
    gsl_vector_memcpy(v0, A->head->data->v);
    free(list_free(A));
    printf("got initial solution via truncated search\n");
    env->params->greer_params.obj_code = WRC;
  }

  // normalize:
  gsl_vector_scale(v0, 1 / gsl_blas_dnrm2(v0));
  class_res res = classify(env, v0);
  printf("Initial solution has tpos = %d, fpos = %d, tneg = %d, fneg = %d, pz "
         "= %d, nz = %d\n",
         res.ntpos, res.nfpos, res.ntneg, res.nfneg, res.npz, res.nnz);
  printf("Objective = %g\n", res.obj);
  printf("v0 = ");
  print_vec(v0);

  /*successive_trunc(vs, v0, 5, 5, 1000);
    exit(0);*/

  hplane_list *A;
  switch (env->params->greer_params.method) {
  case 0:
    A = gr_explore(env, v0, create_list());
    break;
  case 1:
    A = mcts(env, v0, create_list());
    break;
  case 2:
    A = beam_search(env, v0, create_list());
    break;
  case 3:
    A = prec_beam_search(env, v0, create_list());
  default:
    printf("unrecognized method parameter\n");
    return NULL;
  }

  printf("found %d solutions, with obj = %d\n", A->count, A->max_obj);
  printf("iterating all solutions:\n");
  print_hplane_list(A);

  vsamples_free(vs);

  if (A->count == 0)
    return NULL;

  gsl_vector *w = A->head->data->v;
  printf("first solution (obj = %g): ", A->head->data->res.obj);
  print_vec(w);

  double *h = CALLOC(d, double);

  for (size_t i = 0; i < d; i++)
    h[i] = gsl_vector_get(w, i);

  // gsl_vector_free(w); //this will automatically be done by list_free
  free(list_free(A));
  // free_samples_cuda();

  if (env->params->greer_params.bnb == 1 ||
      env->params->greer_params.use_rel == 1)
    GRBfreemodel(gr_model);

  return h;
}
