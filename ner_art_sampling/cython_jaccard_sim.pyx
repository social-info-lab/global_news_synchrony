# the cython implementation of jaccard similairty
# to track interaction with python and bottleneck of performance: cython -a cython_jaccard_sim.pyx


# _jaccard_similarity and _jaccard_similarity2 take the 2-dimensional name entity lists as input
# _jaccard_similarity3 take sorted tuples as input

cdef struct jaccard_return:
	float similarity
	float size_union

# Accurate to 8 decimal places
cdef jaccard_return _jaccard_similarity(tuple vec1, tuple vec2):
	# vec1 and vec2 are sorted tuples
	cdef int length1 = len(vec1)
	cdef int length2 = len(vec2)
	cdef jaccard_return jaccard
	cdef jaccard_return *pjaccard_return = &jaccard
	# count the sum of the numbers of duplicates in both lists
	cdef int repeat_times = 0
	# there are no duplicates, because we use Counter in `create_index`
	intersection = list()
	cdef int index1 = 0
	cdef int index2 = 0
	cdef int num1
	cdef int num2
	while index1 < length1 and index2 < length2:
		num1 = vec1[index1][0]
		num2 = vec2[index2][0]
		if num1 == num2:
			# # ensure we only count a unique element once
			# if not intersection or num1 != intersection[-1]:
			intersection.append(num1)

			index1 = index1 + 1
			index2 = index2 + 1
		elif num1 < num2:
			index1 = index1 + 1
		else:
			index2 = index2 + 1

	cdef float len_intersection = len(intersection)
	cdef float len_union = length1 + length2 - repeat_times - len_intersection

	if len_union:
		pjaccard_return.similarity = float(len_intersection)/float(len_union)
		pjaccard_return.size_union = len_union
	else:
		pjaccard_return.similarity = 0
		pjaccard_return.size_union = 0
	return pjaccard_return[0]


# Accurate to 8 decimal places
cdef jaccard_return _jaccard_similarity2(tuple vec1, tuple vec2):
	# vec1 and vec2 are sorted tuples
	cdef int length1 = len(vec1)
	cdef int length2 = len(vec2)
	cdef jaccard_return jaccard
	cdef jaccard_return *pjaccard_return = &jaccard
	# count the sum of the numbers of duplicates in both lists
	cdef int repeat_times = 0
	# there are no duplicates, because we use Counter in `create_index`
	cdef int len_intersection = 0
	cdef int len_union = 0
	cdef int index1 = 0
	cdef int index2 = 0
	cdef int num1
	cdef int num2
	while index1 < length1 and index2 < length2:
		num1 = vec1[index1][0]
		num2 = vec2[index2][0]
		count1 = vec1[index1][1]
		count2 = vec2[index2][1]

		if num1 == num2:
			# # ensure we only count a unique element once
			# if not intersection or num1 != intersection[-1]:
			len_intersection += vec1[index1][1]
			len_intersection += vec2[index2][1]
			index1 = index1 + 1
			index2 = index2 + 1
		elif num1 < num2:
			index1 = index1 + 1
		else:
			index2 = index2 + 1
	for i in range(length1):
		len_union += vec1[i][1]
	for i in range(length2):
		len_union += vec2[i][1]


	if len_union:
		pjaccard_return.similarity = float(len_intersection)/float(len_union)
		pjaccard_return.size_union = len_union
	else:
		pjaccard_return.similarity = 0
		pjaccard_return.size_union = 0
	return pjaccard_return[0]

# Accurate to 8 decimal places
cdef jaccard_return _jaccard_similarity3(tuple vec1, tuple vec2):
	# vec1 and vec2 are sorted tuples
	cdef int length1 = len(vec1)
	cdef int length2 = len(vec2)
	cdef jaccard_return jaccard
	cdef jaccard_return *pjaccard_return = &jaccard
	# count the sum of the numbers of duplicates in both lists
	cdef int repeat_times = 0
	# there are no duplicates, because we use Counter in `create_index`
	intersection = list()
	cdef int index1 = 0
	cdef int index2 = 0
	cdef int num1
	cdef int num2
	while index1 < length1 and index2 < length2:
		num1 = vec1[index1]
		num2 = vec2[index2]
		if num1 == num2:
			# # ensure we only count a unique element once
			# if not intersection or num1 != intersection[-1]:
			intersection.append(num1)

			index1 = index1 + 1
			index2 = index2 + 1
		elif num1 < num2:
			index1 = index1 + 1
		else:
			index2 = index2 + 1

	cdef float len_intersection = len(intersection)
	cdef float len_union = length1 + length2 - repeat_times - len_intersection

	if len_union:
		pjaccard_return.similarity = float(len_intersection)/float(len_union)
		pjaccard_return.size_union = len_union
	else:
		pjaccard_return.similarity = 0
		pjaccard_return.size_union = 0
	return pjaccard_return[0]

def cython_jaccard_similarity(vec1, vec2):
	return _jaccard_similarity(vec1, vec2)

def cython_jaccard_similarity2(vec1, vec2):
	return _jaccard_similarity2(vec1, vec2)

def cython_jaccard_similarity3(vec1, vec2):
	return _jaccard_similarity3(vec1, vec2)