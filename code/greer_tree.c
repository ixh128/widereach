#include "widereach.h"
#include "helper.h"

#include <math.h>
#include <memory.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_linalg.h>

#define DEBUG_GR 1

params_t *envparams;

tnodestack *create_stack() {
  tnodestack *out = CALLOC(1, tnodestack);
  out->max = 16;
  out->nodes = CALLOC(out->max, tnode *);
  out->top = -1;
  return out;
}

void stack_push(tnodestack *s, tnode *node) {
  if(s->top >= s->max - 1) {
    //if max is reached, double the size
    s->nodes = realloc(s->nodes, 2*s->max*sizeof(tnode *));
    if(!s) {
      printf("failed to realloc stack, capping at %d\n", s->max);
    } else {
      s->max *= 2;
    }
  }
  s->nodes[++s->top] = node;
}

tnode *stack_pop(tnodestack *s) {
  return s->nodes[s->top--];
}

int stack_empty(tnodestack *s) {
  return s->top == -1;
}

tnodestack *stack_free(tnodestack *s) {
  free(s->nodes);
  return s;
}

hplane_list *create_list() {
  hplane_list *l = CALLOC(1, hplane_list); //this will set everything to 0/NULL already
  return l;
}

lnode *make_lnode(hplane_data *item) {
  lnode *out = CALLOC(1, lnode);
  out->data = item;
  return out;
}

void list_add(hplane_list *l, hplane_data *item) {
  lnode *node = make_lnode(item);
  if(!l->head) {
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

void free_class_res(class_res res) {
  free(res.tpos); free(res.fpos); free(res.tneg); free(res.fneg); free(res.pz); free(res.nz);
}

void try_free_data(hplane_data *data) {
  //frees the hplane data if extra and res_ent allow it
  if(data->extra) {
    gsl_vector_free(data->v);
    if(!data->res_ent)
      free_class_res(data->res);
    free(data);
  }
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
  memcpy(cp->res.tpos, data->res.tpos, cp->res.ntpos*sizeof(int));
  memcpy(cp->res.fpos, data->res.fpos, cp->res.nfpos*sizeof(int));
  memcpy(cp->res.tneg, data->res.tneg, cp->res.ntneg*sizeof(int));
  memcpy(cp->res.fneg, data->res.fneg, cp->res.nfneg*sizeof(int));
  memcpy(cp->res.pz, data->res.pz, cp->res.npz*sizeof(int));
  memcpy(cp->res.nz, data->res.nz, cp->res.nnz*sizeof(int));
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
  while(curr) {
    prev = curr;
    curr = curr->next;
    prev->data->lists--;
    try_free_data(prev->data);
    free(prev);
  }
  return l;
}

void list_remove_all(hplane_list *l) {
  //removes everything from l
  list_free(l);
  l->head = NULL;
  l->count = 0;
  l->poison = 1;
}

void tree_remove_node(tnode *node) {
  //removes a node from the tree. frees nothing
  if(node->l_sib) {
    node->l_sib->r_sib = node->r_sib;
  } else {
    //then the node must be the left child or the root
    if(node->parent) {
      assert(node->parent->l_child == node);
      node->parent->l_child = node->r_sib;
    }
  }
  if(node->r_sib) {
    node->r_sib->l_sib = node->l_sib;
  } else {
    //then the node must be the right child or the root
    if(node->parent) {
      assert(node->parent->r_child == node);
      node->parent->r_child = node->l_sib;
    }
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
  for(int i = 0; i < x->size; i++) {
    printf("%g%s", gsl_vector_get(x, i), i == x->size - 1 ? "]\n" : " ");
  }
}

void print_hplane_list(hplane_list *l) {
  if(!l) return;
  lnode *curr = l->head;
  int i = 0;
  while(curr) {
    printf("entry %d: ", i++);
    print_vec(curr->data->v);
    printf("  obj = %g\n", curr->data->res.obj);
    curr = curr->next;
  }
}

void compute_res_obj(vsamples_t *vs, class_res *res) {
  double violation = (envparams->theta - 1)*res->ntpos + envparams->theta*res->nfpos;

  if(violation < 0)
    res->obj = res->ntpos;
  else
    res->obj = -1; //infeas <=> neg objective

  //to compute max possible objective:
  int max_ntpos = res->ntpos + res->npz;
  double min_viol = (envparams->theta - 1)*res->ntpos + envparams->theta*res->nfpos;
  if(min_viol < 0)
    res->max_poss_obj = max_ntpos;
  else
    res->max_poss_obj = -1;

  /*if(res->max_poss_obj > 0)
    printf("max obj = %g\n", res->max_poss_obj);*/

  /*if(violation < 0)
    violation = 0;
  
    res->obj = res->ntpos - violation * envparams->lambda;*/
  
}

class_res classify(vsamples_t *vs, gsl_vector *w) {
  //classify the samples according to w
  //this could be made faster by a constant factor by not repeating the loops, but then tpos, etc would need to be defined as vectors instead of arrays
  assert(vs->class_cnt == 2);
  int *pos_res = CALLOC(vs->count[1], int);
  int *neg_res = CALLOC(vs->count[0], int);
  class_res out = {0};

  if(DEBUG_GR) {printf("classifying w = "); print_vec(w); }
  for(size_t i = 0; i < vs->count[1]; i++) {
    //positive samples
    double x;
    gsl_blas_ddot(vs->samples[1][i], w, &x);
    if(fabs(x) < 1e-10) {
      //      printf("positive-class zero %ld\n", i);
      out.npz++;
      pos_res[i] = 0;
    } else if(x > 0) {
      //printf("true positive %ld\n", i);
      out.ntpos++;
      pos_res[i] = 1;
    } else {
      //printf("false negative %ld, w . s = %g\n", i, x);
      out.nfneg++;
      pos_res[i] = -1;
    }
  }
  for(size_t i = 0; i < vs->count[0]; i++) {
    //negative samples
    double x;
    gsl_blas_ddot(vs->samples[0][i], w, &x);
    if(fabs(x) < 1e-10) {
      //printf("negative-class zero %ld\n", i);
      out.nnz++;
      neg_res[i] = 0;
    } else if(x > 0) {
      //printf("false positive %ld\n", i);
      out.nfpos++;
      neg_res[i] = 1;
    } else {
      //printf("true negative %ld\n", i);
      out.ntneg++;
      neg_res[i] = -1;
    }
  }

  if(DEBUG_GR) printf("results: tpos = %d, fpos = %d, tneg = %d, fneg = %d, pz = %d, nz = %d\n", out.ntpos, out.nfpos, out.ntneg, out.nfneg, out.npz, out.nnz);
  
  out.tpos = CALLOC(out.ntpos, int); int ktpos = 0;
  out.fpos = CALLOC(out.nfpos, int); int kfpos = 0;
  out.tneg = CALLOC(out.ntneg, int); int ktneg = 0;
  out.fneg = CALLOC(out.nfneg, int); int kfneg = 0;
  out.pz = CALLOC(out.npz, int); int kpz = 0;
  out.nz = CALLOC(out.nnz, int); int knz = 0;
  
  for(int i = 0; i < vs->count[1]; i++) {
    if(pos_res[i] == 1) out.tpos[ktpos++] = i;
    else if(pos_res[i] == -1) out.fneg[kfneg++] = i;
    else out.pz[kpz++] = i;
  }
  for(int i = 0; i < vs->count[0]; i++) {
    if(neg_res[i] == 1){
      out.fpos[kfpos++] = i;
    }
    else if(neg_res[i] == -1) out.tneg[ktneg++] = i;
    else out.nz[knz++] = i;
  }

  free(pos_res); free(neg_res);

  out.nwrong = out.nfneg + out.nfpos;
  out.nright = out.ntpos + out.ntneg;

  compute_res_obj(vs, &out);
  
  return out;
}

tnode *free_subtree(tnode *root) {
  //frees the subtree starting at node root, including all associated vectors (but not samples), not including the root node itself (since root is stack-alloc'd)
  //recursive post-order traversal
  static int nodes_freed = 0;
  static int nodes_not_freed = 0;
  tnode *next;
  while(root->l_child) {
    next = root->l_child->r_sib;
    free(free_subtree(root->l_child));
    root->l_child = next;
  }
  //then free root itself
  if(root->data->lists <= 0) {
    if(root->data->lists < 0) printf("error counting lists\n");
    gsl_vector_free(root->data->v);
    free_class_res(root->data->res);
    free(root->data);
    nodes_freed++;
  } else {
    //printf("not freeing a node\n");
    nodes_not_freed++;
  }

  //printf("now freed %d nodes, ignored %d\n", nodes_freed, nodes_not_freed);
  return root;
}

double dot(gsl_vector *x, gsl_vector *y) {
  double out;
  gsl_blas_ddot(x, y, &out);
  return out;
}

gsl_vector *obtain_perp(vsamples_t *vs, tnode *node) {
  //obtain a vector orthogonal to all of the samples in node and its ancestors
  //this does a QR factorization - maybe it's faster to do Gram-Schmidt or something else?
  //pretty similar to get_proj_basis - could be refactored

  size_t d = vs->dimension;
  gsl_matrix *A = gsl_matrix_calloc(d, node->depth);
  int k = 0; //= # columns added to A
  tnode *curr = node;
  int expected_cols = node->depth;

  while(curr->depth > 0) {
    //traverse up the tree and grab the sample from every node except the root
    sample_locator_t loc = curr->loc;
    gsl_matrix_set_col(A, k, vs->samples[loc.class][loc.index]);
    curr = curr->parent;
    k++;
  }
  if(k != expected_cols) {
    printf("[ERR] column error\n");
  }

  /*gsl_matrix *Acopy = gsl_matrix_alloc(d, k); //can be removed
    gsl_matrix_memcpy(Acopy, A);*/

  gsl_matrix *T = gsl_matrix_alloc(k, k);
  gsl_linalg_QR_decomp_r(A, T);
  //final column of Q is the sole null-space vector of A^T, assuming A had full rank
  gsl_matrix *Q = gsl_matrix_alloc(d, d);
  gsl_matrix *R = gsl_matrix_alloc(k, k);
  gsl_linalg_QR_unpack_r(A, T, Q, R);

  gsl_vector *v = gsl_vector_alloc(d);
  gsl_matrix_get_col(v, Q, d-1);

  double dot_test = dot(v, vs->samples[node->loc.class][node->loc.index]); //can be removed
  if(fabs(dot_test) >= 1e-10) {
    printf("[ERR] in obtain_perp: dot = %g\n", dot_test);
    printf("k = %d\n", k);

    gsl_vector *b = gsl_vector_alloc(k);
    printf("A = %ld x %ld\n", A->size1, A->size2);
    printf("d = %ld\n", d);
    gsl_blas_dgemv(CblasTrans, 1, A, v, 0, b);
    printf("\nb = ");
    for(int i = 0; i < k; i++)
      printf("%g%s", gsl_vector_get(b, i), i == k - 1 ? "\n" : " ");
    
    exit(1);
  }

  gsl_matrix_free(A);
  gsl_matrix_free(T);
  gsl_matrix_free(Q);
  gsl_matrix_free(R);

  return v;
}

/*gsl_vector *obtain_perp_svd(vsamples_t *vs, tnode *node) {
  //obtain a vector orthogonal to all of the samples in node and its ancestors
  //this uses SVD, which is slower than QR factorization, but QR wasn't working
  
  size_t d = vs->dimension;
  gsl_matrix *A = gsl_matrix_calloc(d, node->depth);
  int k = 0; //= # columns added to A
  int expected_cols = node->depth;

  while(node->depth > 0) {
    //traverse up the tree and grab the sample from every node except the root
    sample_locator_t loc = node->loc;
    gsl_matrix_set_col(A, k, vs->samples[loc.class][loc.index]);
    node = node->parent;
    k++;
  }
  if(k != expected_cols) {
    printf("[ERR] column error\n");
  }

  gsl_vector *s = gsl_vector_alloc(k);
  gsl_matrix *V = gsl_matrix_alloc(k, k);
  gsl_vector *work = gsl_vector_alloc(k);

  gsl_matrix *Acopy = gsl_matrix_alloc(d, k); //can be removed
  gsl_matrix_memcpy(Acopy, A);
  
  gsl_linalg_SV_decomp(A, V, s, work);
  //assuming that the null space of A^T is nontrivial, the latter columns of U should span it
  //calling SVD replaces A with U
  gsl_vector *v = gsl_vector_alloc(d);
  gsl_matrix_get_col(v, A, k-1); 

  if(dot(v, node->v) != 0) {
    printf("[ERR] in obtain_perp: dot = %g\n", dot(v, node->v));
    
    exit(1);
  }

  gsl_matrix_free(V);
  gsl_matrix_free(A);
  gsl_vector_free(s);
  gsl_vector_free(work);

  return v;
}*/


tnode *new_empty_child(tnode *node) {
  //creates a new empty child of node and returns a pointer to it
  //children are filled in left->right
  tnode *new = CALLOC(1, tnode);
  new->parent = node;
  new->data = make_hplane_data(NULL, (class_res) {0});
  if(!node->l_child) {
    assert(!node->r_child);
    node->l_child = new;
    node->r_child = new;
  } else {
    if(node->r_child->r_sib) {
      printf("[ERR] right child already has a right sibling\n");
    }
    node->r_child->r_sib = new;
    new->l_sib = node->r_child;
    node->r_child = new;
  }
  node->n_children++;
  return new;
}

void create_node_child(vsamples_t *vs, tnode *node, sample_locator_t loc) {
  //creates a new child of node with the given loc and computes its vector
  static int curr_key = 1;
  tnode *child = new_empty_child(node);
  child->loc = loc;
  child->depth = node->depth + 1;
  child->key = curr_key++;
  if(DEBUG_GR) {
    printf("Creating child with id %d at depth %d, representing sample (%d, %ld)\n", child->key, child->depth, loc.class, loc.index);
    printf("  Parent id = %d (with sample (%d, %ld))\n", node->key, node->loc.class, node->loc.index);
  }
  child->data->v = obtain_perp(vs, child);
  if(DEBUG_GR) {printf("  Vector: "); print_vec(child->data->v);}
}

void create_node_children(vsamples_t *vs, tnode *node) {
  //creates the children of node. assuming node has a vector and a classification result
  //each child has a vector orthogonal to each sample corresponding to one of its ancestors (except the root, which has no sample)
  //each child will store the resulting vector, along with a locator to the wrongly-classified sample (according to node->v's classification)
  //if the node is already at depth d-1, no children are created

  if(node->depth >= vs->dimension) {
    printf("attempted to add nodes beyond max depth\n");
    return;
  }

  if(node->depth == vs->dimension - 1) {
    if(DEBUG_GR) printf("node %d is at max depth, no children added\n", node->key);
    return;
  }

  if(DEBUG_GR) printf("creating children of node %d\n", node->key);

  //need to add children only corresponding to samples which come after node's sample
  //ordering the samples, considering positives first, then negatives

  
  for(size_t i = 0; i < node->data->res.nfneg; i++) {
    //positive samples classified wrong
    sample_locator_t loc = (sample_locator_t) {.class = 1, .index = node->data->res.fneg[i]};
    create_node_child(vs, node, loc);
  }
  for(size_t i = 0; i < node->data->res.nfpos; i++) {
    //negative samples classified wrong
    sample_locator_t loc = (sample_locator_t) {.class = 0, .index = node->data->res.fpos[i]};
    create_node_child(vs, node, loc);
  }  
}

void add_children_to_stack(tnodestack *s, tnode *node) {
  //adds all of node's children to s, in order left -> right
  tnode *curr = node->l_child;
  while(curr) {
    stack_push(s, curr);
    curr = curr->r_sib;
  }
}

void flip_res(vsamples_t *vs, class_res *res) {
  //replaces res with what it would be if the hyperplane were flipped
  //fneg <-> tpos, fpos <-> tneg, zeros stay the same
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

  compute_res_obj(vs, res);
}

void check_flip(vsamples_t *vs, tnode *node) {
  //checks if a node would be better off with its vector flipped
  //if so, flips it and modifies the classification accordingly
  class_res res = node->data->res;

  //compute what the viol and obj would be after flipping, to decide whether to flip
  double flip_viol = (envparams->theta - 1)*res.nfneg + envparams->theta*res.ntneg;
  double flip_obj;
  if(flip_viol < 0)
    flip_obj = res.nfneg;
  else
    flip_obj = -1;
  //if(node->res.nfneg + node->res.nfpos > node->res.ntpos + node->res.ntneg) {
  if(flip_obj > res.obj){
    if(DEBUG_GR) printf("flipping node %d\n", node->key);
    //in this case, we'd get a better result by using -v instead of v
    gsl_vector_scale(node->data->v, -1);
    //no need to classify again; the result will be the same, just flipped
    flip_res(vs, &node->data->res);
  }
}

void list_remove(hplane_list *l, lnode *node) {
  if(node == l->head) {
    l->head = node->next;
    if(l->head)
      l->head->prev = NULL;
  } else if (!node->next) {
    node->prev->next = NULL;
  } else {
    node->prev->next = node->next;
    node->next->prev = node->prev;
  }
  node->data->lists--;
  try_free_data(node->data);
}

void list_prune(hplane_list *B, int min_obj) {
  //removes everything in list with a lower max possible objective than min_obj
  lnode *curr = B->head;
  lnode *next;
  while(curr) {
    next = curr->next;
    int max_curr_obj = curr->data->res.obj + curr->data->res.npz;
    if(max_curr_obj < min_obj) {
      //remove curr from B
      list_remove(B, curr);
      free(curr);
      B->count--;
    }
    curr = next;
  }
  B->poison = 1;
}

void list_compute_range(hplane_list *l) {
  if(!l->poison)
    return;

  lnode *curr = l->head;
  int max_poss = curr->data->res.max_poss_obj, min_poss = curr->data->res.obj; //disagrees with greer
  int max_obj = curr->data->res.obj;
  curr = curr->next;
  while(curr) {
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

void gr_update_b(vsamples_t *vs, hplane_data *vd, hplane_list *B) {
  //compares the hyperplane vd to those in B, and updates B to contain all hyperplanes capable of reaching the max known objective

  //TODO: this assertion gets violated
  //assert(vd->lists == 0);
  
  if(DEBUG_GR) {printf("updating B with vector (obj = %g): ", vd->res.obj); print_vec(vd->v);}

  int max_v_obj = vd->res.max_poss_obj;
  int min_v_obj = vd->res.obj; //disagrees with greer
  list_compute_range(B);
  //TODO: this doesn't match exactly what greer says, because his notation doesn't make sense
  if(DEBUG_GR)
    if(max_v_obj > 0) 
      printf("max obj = %d\n", max_v_obj);
  if(max_v_obj >= B->min_possible_obj) {
    //in this case, v can potentially be better than anything in B, so add it
    //first, we remove anything that v will always be better than
    list_prune(B, min_v_obj);
    //then add v to B
    list_add_copy(B, vd);
    //then recompute range (since some elements were removed)
    list_compute_range(B);
  }
  try_free_data(vd);
}

void update_b_flips(vsamples_t *vs, tnode *curr, hplane_list *B) {
  //considers v and -v (where v = curr->v), and adds them as appropriate to B via update_b
  class_res res_p = curr->data->res, res_n;
  //memcpy(&res_n, &res_p, sizeof(class_res));
  //note - these two are "entangled" - their arrays are the same, so freeing one will free the other
  // ^ not anymore (hopefully)
  hplane_data *n_data = dup_hplane_data(curr->data);
  flip_res(vs, &n_data->res);
  n_data->extra = 1;
  
  if(res_p.obj > res_n.obj) {
    gr_update_b(vs, curr->data, B);
    try_free_data(n_data);
  } else if(res_p.obj < res_n.obj) {
    gsl_vector_scale(n_data->v, -1);
    gr_update_b(vs, n_data, B);
  } else {
    /*gsl_vector *v_n = gsl_vector_alloc(curr->data->v->size);
    gsl_vector_memcpy(v_n, curr->data->v);
    gsl_vector_scale(v_n, -1);
    hplane_data *n_data = make_hplane_data(v_n, res_n);
    n_data->extra = 1;
    curr->data->res_ent = n_data->res_ent = 1;*/

    gsl_vector_scale(n_data->v, -1);    
    gr_update_b(vs, curr->data, B);
    gr_update_b(vs, n_data, B);
  }
}

void gr_update_a(hplane_data *vd, hplane_list *A) {
  if(DEBUG_GR) {printf("updating A with vector (obj = %g): ", vd->res.obj); print_vec(vd->v);}
  list_compute_range(A);
  if(A->count == 0) {
    list_add_copy(A, vd);
  } else {
    if(vd->res.obj >= A->max_obj) {
      if(vd->res.obj > A->max_obj) {
	list_remove_all(A);
      }
      list_add_copy(A, vd);
    }
    try_free_data(vd);
  }
  list_compute_range(A);
}

int dim_zeros(vsamples_t *vs, hplane_data *vd) {
  //computes the dimension of the span of the samples lying on v
  //I'll do this by making them the columns of a matrix, then computing its rank via SVD (there is probably a faster way than SVD, but it's a small matrix)
  int ncols = vd->res.npz + vd->res.nnz;
  if(ncols == 0) return 0;

  size_t d = vs->dimension;
  gsl_matrix *A = gsl_matrix_calloc(d, ncols);
  int k = 0; //current column index
  //positive samples:
  for(int i = 0; i < vd->res.npz; i++) {
    gsl_matrix_set_col(A, k++, vs->samples[1][vd->res.pz[i]]);
  }
  //negative samples
  for(int i = 0; i < vd->res.nnz; i++) {
    gsl_matrix_set_col(A, k++, vs->samples[0][vd->res.nz[i]]);
  }
  if(k != ncols) {
    printf("k != ncols in dim_zeros (%d != %d)\n", k, ncols);
    exit(1); //TODO - remove
  }

  gsl_vector *s = gsl_vector_alloc(ncols);
  gsl_matrix *V = gsl_matrix_alloc(ncols, ncols);
  gsl_vector *work = gsl_vector_alloc(ncols);

  gsl_linalg_SV_decomp(A, V, s, work);

  int dim = 0;
  for(int i = 0; i < ncols; i++) {
    double si = gsl_vector_get(s, i);
    if(fabs(si) > 1e-10)
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
  for(int class = 0; class < vs->class_cnt; class++) {
    for(int i = 0; i < vs->count[class]; i++) {
      //positive samples
      double x_dot_y, z_dot_y;
      gsl_blas_ddot(x, vs->samples[class][i], &x_dot_y);
      if(fabs(x_dot_y) < 1e-10) //then y is on the hyperplane, so ignore it
	continue;
      
      gsl_blas_ddot(z, vs->samples[class][i], &z_dot_y);

      if((x_dot_y > 0 && z_dot_y < 0) || (x_dot_y < 0 && z_dot_y > 0)) {
	min = fmin(min, x_dot_y/(x_dot_y - z_dot_y));
      }
    }
  }
  return min/2;
}

void update_a_displace(vsamples_t *vs, hplane_data *vd, gsl_vector *y, hplane_list *A) {
  //helper function for solve_dim1 and solve_pc, where we have to displace vd.v towards y, then add it to A
  double alpha = comp_disp(vs, vd->v, y);
  gsl_vector *displaced = gsl_vector_alloc(y->size);
  gsl_vector_memcpy(displaced, vd->v);
  if(DEBUG_GR) {
    printf("alpha = %g\n", alpha);
    printf("v = "); print_vec(vd->v);
    printf("y = "); print_vec(y);
  }
  gsl_vector_axpby(alpha, y, 1-alpha, displaced); //v <- alpha y + (1-alpha)v
  
  //TODO: possibly scale the displaced vector, since it won't have unit norm anymore
  hplane_data *displaced_data = make_hplane_data(displaced, classify(vs, displaced));
  displaced_data->extra = 0;
  gr_update_a(displaced_data, A);
  gsl_vector_free(displaced);
  free_class_res(displaced_data->res);
  free(displaced_data);
}

void solve_dim1(vsamples_t *vs, hplane_data *vd, hplane_list *A) {
  gsl_vector *y;
  //y needs to be assigned to something on the hplane
  //preferring positive samples here, but I don't know if that's best
  assert(vd->res.npz + vd->res.nnz > 0);
  //if y is a negative sample, need to copy it and flip it
  int class_label;
  if(vd->res.npz > 0) {
    y = vs->samples[1][vd->res.pz[0]];
    class_label = 1;
  } else {
    y = gsl_vector_alloc(vd->v->size);
    gsl_vector_memcpy(y, vs->samples[0][vd->res.nz[0]]);
    gsl_vector_scale(y, -1);
    class_label = -1;
  }

  update_a_displace(vs, vd, y, A);

  if(class_label == -1)
    gsl_vector_free(y);

  //TODO (important): if there is a sample equal to -y, need to do extra
  //but that won't likely come up in real datasets
  //see greer book pg 104 bottom for how to deal with it
}

void solve_pc(vsamples_t *vs, hplane_data *vd, gsl_vector *x, hplane_list *A) {
  update_a_displace(vs, vd, x, A);
}

void solve_highdim(vsamples_t *vs, hplane_data *vd, hplane_list *A, int dim) {
  if(DEBUG_GR) {printf("solving LP, starting with the vector (obj = %g): ", vd->res.obj); print_vec(vd->v);}
  pc_soln sol = detect_pointed_cone(vs, *vd);
  double gamma = sol.gamma;
  gsl_vector *x = sol.x;
  if(DEBUG_GR) {printf("result: gamma = %g, x = ", gamma); print_vec(x);}
  assert(gamma == 0 || gamma == 1);

  switch((int) gamma) {
  case 1:
    solve_pc(vs, vd, x, A);
    break;
  case 0:
    printf("recursive call needed, exiting...\n");
    exit(1);
    break;
  default:
    printf("[ERR] gamma = %g\n", gamma);
    exit(1);
  }

  gsl_vector_free(x);
}

hplane_list *gr_displace(vsamples_t *vs, hplane_list *B, hplane_list *A) {
  if(DEBUG_GR) {printf("entered displace. B = \n"); print_hplane_list(B);}

  lnode *curr = B->head, *next;
  while(curr) {
    int sub_dim = dim_zeros(vs, curr->data);
    if(DEBUG_GR) printf("sub_dim = %d\n", sub_dim);

    curr->data->extra = 1; //allow update_a to free curr
    switch(sub_dim) {
    case 0:
      gr_update_a(curr->data, A);
      break;
    case 1:
      solve_dim1(vs, curr->data, A);
      try_free_data(curr->data);
      break;
    default:
      solve_highdim(vs, curr->data, A, sub_dim);
      try_free_data(curr->data);
    }
    next = curr->next;
    free(curr);
    curr = next;
  }
  return A;
}

void replace_with_copies(hplane_list *l) {
  //replaces every non-extra vector in l with a copy, so that the tree can be freed
  //also copies the classification results for the same reason
  lnode *curr = l->head;
  while(curr) {
    if(!curr->data->extra) {
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
  //set all entries in l to be "extra"
  lnode *curr = l->head;
  while(curr) {
    curr->data->extra = 1;
    curr = curr->next;
  }
}

void check_free(tnode *node) {
  //given a max-depth node, frees as much of its subtree as possible

  tnode *curr = node;
  while(curr->parent && curr->parent->n_children == curr->parent->children_explored && curr->parent->depth > 0) {
    curr = curr->parent;
  }

  tree_remove_node(curr);
  free(free_subtree(curr));
}

hplane_list *gr_explore(vsamples_t *vs, gsl_vector *v0, hplane_list *A) {
  /*class_res res = classify(vs, v0);
  if(res.nfneg + res.nfpos > res.ntpos + res.ntneg) {
    gsl_vector_scale(v0, -1);
    res = classify(vs, v0); //TODO: this is unnecessary - just rearrange the existing items in res
  }
  if(res.nfneg + res.nfpos == 0) {
    printf("perfect vector found\n");
    return; //TODO - call displace
    }*/
  //actually, everything above here seems unnecessary - it could just be done in the DFS below

  size_t d = vs->dimension;
  class_res root_res = classify(vs, v0);
  tnode root = (tnode) {make_hplane_data(v0, root_res), .key=0}; //set vector v0 and depth = 0, nothing else
  if(DEBUG_GR) {printf("v0 = "); print_vec(v0);}

  tnodestack *s = create_stack();
  stack_push(s, &root);

  double best_obj = -1e101; //temporary - no need to fill in now
  gsl_vector *best_v = v0;
  hplane_list *B = create_list();
  list_add_copy(B, root.data);
  free_class_res(root_res);

  int nodes_explored = 0;

  //  time_t start = time(0);

  while(!stack_empty(s)) {
    tnode *curr = stack_pop(s);
    if(DEBUG_GR) printf("popped from stack, id = %d\n", curr->key);
    nodes_explored++;
    /*if(time(0) - start > 0) {
      printf("stack size = %d\n", s->top);
      start = time(0);
      }*/

    //find the vector v for curr if it doesn't already exist
    if(!curr->data->v) { //must be the root node in this case
      //actually, this is unncessary, since the root node will already have v0
      printf("[ERR] found node without a vector\n");
      curr->data->v = v0;
    }
    //classify by v and store it in the node
    class_res res = classify(vs, curr->data->v);
    curr->data->res = res;
    check_flip(vs, curr);

    if(curr->data->res.obj > best_obj) {
      best_obj = curr->data->res.obj;
      best_v = curr->data->v;
      printf("new best soln with objective %g\n", best_obj);
    }

    if(res.nwrong == 0) {
      printf("perfect vector found\n");
      list_remove_all(B);
      list_add_copy(B, curr->data);
      gr_displace(vs, B, A);
      return A;
    }
    
    /*if(res.nfneg + res.nfpos + res.nnz + res.npz - curr->depth < root.data->res.nfneg + root.data->res.nfpos) {
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
      printf("restarting, A has %d hplane(s)\n", A->count);
      return gr_explore(vs, new_v0, A);
    }*/
    
    create_node_children(vs, curr);
    if(curr->depth < d - 1) {
      add_children_to_stack(s, curr);
      if(DEBUG_GR) printf("calling update_b\n");
      gr_update_b(vs, curr->data, B);
    } else {
      if(DEBUG_GR) printf("calling update_b_flips\n");
      update_b_flips(vs, curr, B);
    }

    //if possible, we want to free the subtree that we've already explored
    if(curr->parent) {
      curr->parent->children_explored++;
    }
    if(curr->depth == d - 1) {
      check_free(curr);
    }
  }
  //free the tree
  //before that, we need to make sure all of the vectors in B have been copied, so that we don't free them
  /*printf("replacing B with copies\n");
    replace_with_copies(B);*/

  //now we can free it
  free_subtree(&root);

  free(stack_free(s));

  //exit(0);

  printf("finished tree search, now displacing %d boundary vectors\n", B->count);

  gr_displace(vs, B, A);
  printf("best obj = %d\n", A->max_obj);
  printf("explored %d nodes\n", nodes_explored);

  free(B);
  
  return A;
}

double *single_greer_run(env_t *env, double *h0) {
  vsamples_t *vs = samples_to_vec(env->samples);
  envparams = env->params;
  size_t d = env->samples->dimension;

  double *h;
  if(h0)
    h = h0;
  else
    h = best_random_hyperplane_unbiased(1, env);
  gsl_vector hvec = gsl_vector_view_array(h, d).vector;

  gsl_vector *v0 = gsl_vector_calloc(vs->dimension);
  //gsl_vector_set(v0, 0, 1);
  gsl_vector_memcpy(v0, &hvec);

  //normalize:
  gsl_vector_scale(v0, 1/gsl_blas_dnrm2(v0));
  class_res res = classify(vs, v0);
  printf("Initial solution has tpos = %d, fpos = %d, tneg = %d, fneg = %d, pz = %d, nz = %d\n", res.ntpos, res.nfpos, res.ntneg, res.nfneg, res.npz, res.nnz);

  hplane_list *A = gr_explore(vs, v0, create_list());

  printf("found %d solutions, with obj = %d\n", A->count, A->max_obj);
  printf("iterating all solutions:\n");
  print_hplane_list(A);

  vsamples_free(vs);

  if(A->count == 0)
    return NULL;

  gsl_vector *w = A->head->data->v;
  printf("first solution (obj = %g): ", A->head->data->res.obj);
  print_vec(w);

  for(size_t i = 0; i < d; i++)
    h[i] = gsl_vector_get(w, i);
  
  //gsl_vector_free(w); //this will automatically be done by list_free
  free(list_free(A));
  
  return h;
}
