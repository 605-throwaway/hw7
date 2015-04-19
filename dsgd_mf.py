from scipy.sparse import csr_matrix, csc_matrix
from numpy import int8, random, empty, savetxt, hstack, vstack
from pyspark import SparkContext, SparkConf
from sys import argv


def parse_autolab_line(line):
    """
    Parses a line of the input file.
    Returns (user-1, movie-1, rank)) so the lists are zero indexed.
"""
    split_line = line.split(",")
    return (int(split_line[0]) - 1, int(split_line[1]) - 1, int8(split_line[2]))

def get_data_autolab_format(input_v_filepath, rdd):
    """
Reads the input file into an RDD and maps each line to
a key value pair (user, movie, rank)
"""
    exp_data = rdd.textFile(input_v_filepath)
    exp_data.persist()
    return exp_data.map(parse_autolab_line)

def get_v_dims(data):
    """
The argument data is an RDD with  (user-1, movie-1, rank)
This calculates and returns the number of rows (users) and columns (movies)
for when this is mapped to the matrix V.
Since the users and movies are zero indexed, I add one in the return line.
"""
    #num_rows is the number of the highest user number in data.
    num_rows = data.map(lambda x: x[0]).reduce(lambda a, b: max(a, b))

    #num_cols is the number of the highest movie number in data.
    num_cols = data.map(lambda x: x[1]).reduce(lambda a, b: max(a, b))
    return num_rows + 1, num_cols + 1

def update_w_h(block, rows_per_block, cols_per_block, num_workers, lambda1,
               init_iteration, beta):
    """
This does the SGD update on one block of the matrix.
input:  (block, (v, ((w, n), (h, n)) ) )
output: ((w_block, (w, n)), (h_block, (h, n)))
"""
    block_num = block[0]
    v = block[1][0]
    w = block[1][1][0][0]
    wn = block[1][1][0][1]
    h = block[1][1][1][0]
    hn = block[1][1][1][1]

    #init_iteration is the number of iterations before the current strata
    #iterations keeps track of the iteration number within this block
    iteration = init_iteration

    #Updates w and h for each training point in the block of V
    #triple is (user-1, movie-1, rank) from the input data
    for triple in v:

        #This gets the index numbers for w and h for the current data point
        user = triple[0] % rows_per_block
        movie = triple[1] % cols_per_block

        w_row = w[user, :]
        h_col = h[:, movie]
        epsilon = (100 + iteration)**(-beta)

        #This calculates the gradient update for w and h.
        grad_w = (-2 * (triple[2] - w_row.dot(h_col)[0, 0]) * h_col.transpose())
        grad_w += ((2 * lambda1 / wn[user]) * w_row)
        new_w = w_row - (epsilon * grad_w)

        grad_h = (-2 * (triple[2] - w_row.dot(h_col)[0, 0]) * w_row.transpose())
        grad_h += ((2 * lambda1 / hn[movie]) * h_col)
        new_h = h_col - (epsilon * grad_h)

        #This updates w and h and the iteration number.
        w[user, :] = new_w
        h[:, movie] = new_h
        iteration += 1


    #This updates the block numbers for w and h so they get matched with
    #the correct block of v in the next iteration
    w_block = block_num + 1
    h_block = block_num - num_workers
    if w_block % num_workers == 0:
        w_block = w_block - num_workers
    if h_block < 0:
        h_block = num_workers * (num_workers - 1) + (block_num % num_workers)

    return ((w_block, (w, wn)), (h_block, (h, hn)), iteration - init_iteration)

def make_w_n(full_v_blocks, num_workers, rows_per_block, num_rows,
             num_factors, max_block_num):
    """
    #full_v_blocks is (block of V, iter(user, user, rank))
    #This sets the keys to be the block number of the main diagonal of V
    #because that is the first strata
    #Step 1: (user, 1)
    #Step 2: (user, count)
    #Step 3: (block of V, (user, count))
    #Step 4: (block of V, iter(user, count))
    #Step 5: (block_num, (w, n))
    Step 5 is returned.
"""
    user_counts1 = full_v_blocks.flatMap(lambda y: [(x[0], 1) for x in y[1]])
    user_counts1.persist()
    user_counts2 = user_counts1.reduceByKey(lambda a, b: a + b)
    user_counts2.persist()
    user_counts3 = user_counts2.map(lambda x:
                                    ((num_workers + 1) *
                                     (x[0] // rows_per_block), x))
    user_counts3.persist()
    user_counts4 = user_counts3.groupByKey(num_workers)
    user_counts4.persist()
    w_n = user_counts4.map(lambda x: init_w(x, rows_per_block, num_rows,
                                         num_factors, max_block_num))
    w_n.persist()
    return w_n

def init_w(user_block, rows_per_block, num_rows, num_factors, max_block_num):
    """
Initializes blocks of w.  Each block has a w matrix and an n vector.  The n
vector has the total number of ratings for that row of v.  The w matrix has
zero entries for rows where n = 0 and random entries otherwise.
"""
    block_num = user_block[0]
    curr_block_size = rows_per_block

    if block_num == max_block_num:
        curr_block_size = num_rows % rows_per_block


    w = csr_matrix((curr_block_size, num_factors))
    w_row = random.rand(1, num_factors)
    n = empty(curr_block_size, int)

    for user_count in user_block[1]:
        w[user_count[0] % rows_per_block, :] = w_row
        n[user_count[0] % rows_per_block] = user_count[1]

    return (block_num, (w, n))


def make_h_n(full_v_blocks, num_workers, cols_per_block, num_cols,
             num_factors, max_block_num):
    """
    #full_v_blocks is (block of V, iter(user, movie, rank))
    #This sets the keys to be the block number of the main diagonal of V
    #because that is the first strata
    #Step 1: (movie, 1)
    #Step 2: (movie, count)
    #Step 3: (block of V, (movie, count))
    #Step 4: (block of V, iter(movie, count))
    #Step 5: (block_num, (h, n))
    Step 5 is returned.
"""
    movie_counts1 = full_v_blocks.flatMap(lambda y: [(x[1], 1) for x in y[1]])
    movie_counts1.persist()
    movie_counts2 = movie_counts1.reduceByKey(lambda a, b: a + b)
    movie_counts2.persist()
    movie_counts3 = movie_counts2.map(lambda x:
                                      ((num_workers + 1) *
                                       (x[0] // cols_per_block), x))
    movie_counts3.persist()
    movie_counts4 = movie_counts3.groupByKey(num_workers)
    movie_counts4.persist()
    h_n = movie_counts4.map(lambda x: init_h(x, cols_per_block, num_cols,
                                         num_factors, max_block_num))
    h_n.persist()
    return h_n

def init_h(movie_block, cols_per_block, num_cols, num_factors, max_block_num):
    """
Initializes blocks of h.  Each block has a h matrix and an n vector.  The n
vector has the total number of ratings for that column of v.  The h matrix has
zero entries for columns where n = 0 and random entries otherwise.
"""
    block_num = movie_block[0]

    curr_block_size = cols_per_block

    if block_num == max_block_num:
        curr_block_size = num_cols % cols_per_block

    h = csc_matrix((num_factors, curr_block_size))
    h_col = random.rand(num_factors, 1)
    n = empty(curr_block_size, int)

    for movie_count in movie_block[1]:
        h[:, movie_count[0] % cols_per_block] = h_col
        n[movie_count[0] % cols_per_block] = movie_count[1]

    return (block_num, (h, n))

def make_v_blocks(num_workers, full_v_blocks, rdd):
    """
full_v_blocks only has rdd entries for blocks of v with rankings.  This adds
empty blocks so there are no more missing blocks.  This is important because
of the way strata are computed in the sgd step.
"""
    all_block_nums = rdd.parallelize(xrange(num_workers**2))
    all_block_nums.persist()
    non_empty_v_blocks = set(full_v_blocks.map(lambda x: x[0]).collect())
    empty_v_blocks = all_block_nums.filter(lambda x: not x in
                                           non_empty_v_blocks).map(lambda x:
                                                                   (x, []))
    empty_v_blocks.persist()
    v_blocks = full_v_blocks.union(empty_v_blocks)
    v_blocks.persist()
    return v_blocks

def do_sgd(v_blocks, num_iterations, w_n, h_n, rows_per_block, cols_per_block,
           num_workers, lambda_value, beta_value):
    """
Does the inner loop of parallel sgd updates and reassigns strata in order to
do all iterations of the sgd.
Returns an RDD of ((w_block, (w, wn)), (h_block, (h, hn)), iteration)
"""
    sgd_iter = 1
    for iter_num in xrange(num_iterations):
        curr_strata = set(w_n.map(lambda x: x[0]).collect())
        curr_v_blocks = v_blocks.filter(lambda x: x[0] in curr_strata)
        curr_v_blocks.persist()

        #w_h is (block, ((w, n), (h, n)))
        w_h = w_n.join(h_n)
        w_h.persist()

        #v_w_h is (block, (v, ((w, n), (h, n))))
        v_w_h = curr_v_blocks.join(w_h)
        v_w_h.persist()

        #w_h is ((w_block, (w, wn)), (h_block, (h, hn)),
        #iteration number of sgd iterations on the block)
        w_h = v_w_h.map(lambda x: update_w_h(x, rows_per_block, cols_per_block,
                                             num_workers, lambda_value,
                                             sgd_iter, beta_value))
        w_h.persist()

        sgd_iter += sum(w_h.map(lambda x: x[2]).collect())

        #w_n is (w_block, (w, wn)
        w_n = w_h.map(lambda x: x[0])
        w_n.persist()
        #h_n is (h_block, (h, hn)
        h_n = w_h.map(lambda x: x[1])
        h_n.persist()

    return w_h

def write_output_to_file(w_h, output_w_filepath, num_workers,
                         output_h_filepath):
    """
This writes the w and h matrices to their output files.
 w_h is (block, ((w, n), (h, n)))
Block is the block of V so w and h need to be renumbered to put them in the
correct order.
The blocks of w go in numeric order since they correspond to rows of v
The blocks of h go in block % num_workers order since they correspond
to columns of v
ordered_w and ordered_h contain the n vectors.
"""
    ordered_w = w_h.map(lambda x: x[0]).sortByKey().collect()
    full_w = vstack([x[1][0].toarray() for x in ordered_w])
    savetxt(output_w_filepath, full_w, delimiter=',')

    ordered_h = w_h.map(lambda x: (x[1][0] % num_workers,
                                   x[1][1])).sortByKey().collect()
    full_h = hstack([x[1][0].toarray() for x in ordered_h])
    savetxt(output_h_filepath, full_h, delimiter=',')

def main():
    """
    Parses the constants from the command line input.
"""
    num_factors = int(argv[1])
    num_workers = int(argv[2])
    num_iterations = int(argv[3])
    beta_value = float(argv[4])
    lambda_value = float(argv[5])
    input_v_filepath = argv[6]
    output_w_filepath = argv[7]
    output_h_filepath = argv[8]

    conf = SparkConf().setMaster("local[%d]" % num_workers)
    rdd = SparkContext(conf=conf)

    #autolab_triples is an rdd of (user-1, movie-1, rank)
    autolab_triples = get_data_autolab_format(input_v_filepath, rdd)
    autolab_triples.persist()

    #num_rows (or cols) is the number of rows in the v matrix which I never
    #explicitly construct.
    num_rows, num_cols = get_v_dims(autolab_triples)

    #Computes the ceiling of rows or cols / workers
    rows_per_block = -(-num_rows // num_workers)
    cols_per_block = -(-num_cols // num_workers)

    #keyed_autolab_triples are (block, (user-1, movie-1, rank))
    #I "construct" v by dividing it up into (num_workers^2) blocks
    #so the first set of rows is blocks 0, 1, 2...num_workers - 1
    keyed_autolab_triples = autolab_triples.map(lambda x:
                                                (num_workers*(x[0] //
                                                              rows_per_block) +
                                                 (x[1] // cols_per_block), x))

    keyed_autolab_triples.persist()

    #Calculates the max_block_num
    max_block_num = num_workers * ((num_rows - 1) // rows_per_block)
    max_block_num += ((num_cols -1) // cols_per_block)

    #keyed_autolab_triples are (block, iter(user-1, movie-1, rank))
    full_v_blocks = keyed_autolab_triples.groupByKey()
    full_v_blocks.persist()

    #h_n is (h_block, (h, hn))
    #Blocks of h_n are on the diagonal set of blocks of v.
    #h is a matrix with the h values for those columns.
    #hn is a vector with the total number of ratings for the
    #corresponding column.  w_n is similar but for rows of v.
    h_n = make_h_n(full_v_blocks, num_workers, cols_per_block, num_cols,
             num_factors, max_block_num)

    w_n = make_w_n(full_v_blocks, num_workers, rows_per_block, num_rows,
             num_factors, max_block_num)


    v_blocks = make_v_blocks(num_workers, full_v_blocks, rdd)

    w_h = do_sgd(v_blocks, num_iterations, w_n, h_n, rows_per_block,
                 cols_per_block, num_workers, lambda_value, beta_value)

    write_output_to_file(w_h, output_w_filepath, num_workers, output_h_filepath)


if __name__ == "__main__":
    main()


