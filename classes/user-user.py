# This is a class made to reccomend classes using user-user content based reccomendation for specifically the files in this assignment
# Because we use certain labels to acess data like 'StudentId' this is not a general reccomender
# This can be initialized by inputting a dataframe where each row represents a class a student took, and assosiates the students id with the course id for that respective row
class userUserReccomender:
    # Stored values we compute, useful for cacheing values we will use commonly and stops us from recomputing or updating data frames excessively
    input_data: pd.DataFrame
    training_data: pd.DataFrame
    testing_data: pd.DataFrame
    student_course_matrix: pd.DataFrame
    # A boolean we use to indicate if we need to recompute the user-course matrix used for our reccomendation options. This matrix is large, almost half a gigabyte so it
    # is important we only update it when we need to
    matrix_current: bool
    # This is used to partition the data off into training and testing data. We tend not to use this function because it it makes more sense to simply omit certain classes from the
    # Students we want to train on instead of keeping everything seperate
    # This selecteds a proportion of students, provided by arguments to set into a training data set and puts the other students into a testing set
    # Keep in mind this partitions on students, not individual courses so we don't end up with incomplete student carreers in both sets
    def partitionData(self, proportion_training: float) -> None:
        all_student_ids = self.input_data.StudentId.unique()
        training_students = np.random.choice(all_student_ids, int(all_student_ids.size * proportion_training), False)
        testing_students = np.array([i for i in all_student_ids if i not in training_students])
        self.training_data = self.input_data[self.input_data["StudentId"].isin(training_students)]
        self.testing_data = self.input_data[self.input_data["StudentId"].isin(testing_students)]
        self.matrix_current = False
    # Used to undo a partition, by default all values are stored in the input data and this additionally stores them in our testing data
    # This is because our testing data is what is mainly used to compute our matrix
    def dePartitionData(self) -> None:
        self.training_data = self.input_data
        self.testing_data = pd.DataFrame()
        self.matrix_current = False
    # This is called on our input matrix, we first drop all students with all nan values for course score
    # Next we drop all students that have taken under 22 courses, these are people who have either dropped out or are not completed their degrees,
    # this means that the students will not be good to reccomend on because they will only reccomend a small subset of classes
    def preProcessData(self) -> None:
        #We only want data with large enough class counts to support a nuanced reccomendation. 
        self.dePartitionData()
        self.updateMatrix()
        dropIds = self.student_course_matrix[self.student_course_matrix.count(axis=1) < 22].index
        self.input_data = self.input_data.drop(self.input_data[self.input_data.StudentId.isin(dropIds)].index)
        self.dePartitionData()
        self.updateMatrix()
    # The init function takes in a daframe where courses taken by a particular student are rows, we need scores included for our system to work
    def __init__(self, input_data: pd.DataFrame) -> None:
        self.input_data = input_data
        self.preProcessData()
        self.matrix_current = True
    # This recomuputes our user-course matrix, you can do this externally but there are checks to prevent outdated matricies from being used internally
    def updateMatrix(self) -> None:
        self.student_course_matrix = self.training_data.pivot_table(index='StudentId', columns='CourseId', values='Score')
        self.matrix_current = True
    # Calculates reccomended courses for a single user, it outputs three values:
        # The fist value is the input courses, these are a subset of the courses the user we are predicting on took during their carreer
        # The second is the other courses the student took, that we did not use to predict on, we seperate these because our algorithm does not predict on courses we have allready taken, so we have the output courses seperate as what we compare against to evaluate accuracy
        # The third is the predictions we have made, we want these to match as closely as possible to our output courses, and not include any courses from our input courses
    def calculateSingleReccomendation(self, student_id: int, proportion_classes_taken: float, num_simlar_students: int) -> tuple[pd.Series, pd.Series, pd.Series]:
        if not self.matrix_current: self.updateMatrix()
        # Make and store sample carreers for output and updating our matrix
        full_carreer = self.student_course_matrix.loc[student_id].copy(deep=True)
        drop_courses = full_carreer.dropna().sample(int(full_carreer.count() * (1-proportion_classes_taken)))
        sample_carreer = full_carreer.copy(deep=True)
        sample_carreer.loc[drop_courses.index.tolist()] = np.nan
        #We update our matrix so the user selected appears to have only taken the sample carreer courses
        temp_matrix = self.student_course_matrix.copy(deep=True)
        temp_matrix.loc[student_id] = sample_carreer # type: ignore
        #Normalize data by subtracting the gpa
        #temp_matrix = temp_matrix.subtract(temp_matrix.mean(axis=1), axis=0)
        #sample_carreer = sample_carreer.subtract(full_carreer.mean())
        #drop_courses = drop_courses.subtract(full_carreer.mean())
        # Identify similar users by using cosine similarity across our matrix, but only compare on the courses our sample student has taken
        # We do this because if a student has taken many more courses than our sample student their similarity will be lower. So by only comparing to user taken courses we eliminate this issue
        similar_students = pd.DataFrame(cosine_similarity(temp_matrix[sample_carreer.dropna().index].fillna(0)), index=temp_matrix.index, columns=temp_matrix.index).drop(student_id)[student_id].sort_values(ascending=False)[:num_simlar_students]
        taken_courses = temp_matrix[temp_matrix.index == student_id].dropna(axis=1, how='all')
        reccomended_courses = temp_matrix[temp_matrix.index.isin(similar_students.index)].dropna(axis=1, how='all')
        # We drop courses we have allready taken from the reccomendations because reccomending courses we have allready taken makes no sense
        reccomended_courses.drop(taken_courses.columns,axis=1, inplace=True, errors='ignore')
        reccomendations = {}
        #Run through all our possible reccomended scores and predict a score as the weighted average of similar users scores based off similarity
        for course in reccomended_courses.columns:
            course_scores = reccomended_courses[course]
            course_score, course_count = 0,0
            for student in similar_students.index:
                if pd.isna(course_scores[student]): continue
                course_score += course_scores[student] * similar_students[student]
                course_count += similar_students[student]
            reccomendations[course] = course_score/course_count
        reccomendations = pd.Series(reccomendations, index=reccomendations.keys()).sort_values(ascending=False) # type: ignore
        return(sample_carreer, drop_courses, reccomendations)
    # Calculates reccomended courses for a single student provided via input of a series, the series index must include all courses or this will crash, it outputs three values:
        # The fist value is the input courses, these are a subset of the courses the user we are predicting on took during their carreer
        # The second is the other courses the student took, that we did not use to predict on, we seperate these because our algorithm does not predict on courses we have allready taken, so we have the output courses seperate as what we compare against to evaluate accuracy
        # The third is the predictions we have made, we want these to match as closely as possible to our output courses, and not include any courses from our input courses
    def calculateReccomendation(self, courses: pd.Series, student_id: int, num_similar_students: int) -> pd.Series:
        if not self.matrix_current: self.updateMatrix()
        temp_matrix = self.student_course_matrix.copy(deep=True)
        temp_matrix.drop(student_id, inplace=True)
        temp_matrix.loc[student_id] = courses # type: ignore
        #Normalize data by subtracting the gpa
        #temp_matrix = temp_matrix.subtract(temp_matrix.mean(axis=1), axis=0)
        #sample_carreer = sample_carreer.subtract(full_carreer.mean())
        #drop_courses = drop_courses.subtract(full_carreer.mean())
        # Identify similar users by using cosine similarity across our matrix, but only compare on the courses our sample student has taken
        # We do this because if a student has taken many more courses than our sample student their similarity will be lower. So by only comparing to user taken courses we eliminate this issue
        similar_students = pd.DataFrame(cosine_similarity(temp_matrix[courses.dropna().index].fillna(0)), index=temp_matrix.index, columns=temp_matrix.index).drop(student_id)[student_id].sort_values(ascending=False)[:num_similar_students]
        taken_courses = temp_matrix[temp_matrix.index == student_id].dropna(axis=1, how='all')
        reccomended_courses = temp_matrix[temp_matrix.index.isin(similar_students.index)].dropna(axis=1, how='all')
        # We drop courses we have allready taken from the reccomendations because reccomending courses we have allready taken makes no sense
        reccomended_courses.drop(taken_courses.columns,axis=1, inplace=True, errors='ignore')
        reccomendations = {}
        #Run through all our possible reccomended scores and predict a score as the weighted average of similar users scores based off similarity
        for course in reccomended_courses.columns:
            course_scores = reccomended_courses[course]
            course_score, course_count = 0,0
            for student in similar_students.index:
                if pd.isna(course_scores[student]): continue
                course_score += course_scores[student] * similar_students[student]
                course_count += similar_students[student]
            reccomendations[course] = course_score/course_count
        reccomendations = pd.Series(reccomendations, index=reccomendations.keys()).sort_values(ascending=False) # type: ignore
        return reccomendations
    # Same as above but with multiple students, this means we output 2d dataframes where every row represents the following, respectively per dataframe
        # The subset of student classes for that student we are reccomending on
        # The subset of student classes that were not used to generate reccomendations, these are what we compare againt for accuracy
        # The reccomendations for the given student, these will be compared againt output 2 for accuracy
    def calculateMultipleReccomendations(self, student_ids: list[int], proportion_classes_taken: float, num_similar_students: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Update matrix if needed
        if not self.matrix_current: self.updateMatrix()
        #Grab students we want to reccomend on, and generate a list of courses we are dropping from their carrers and the result sample carreer after the drop
        full_carreers = self.student_course_matrix.loc[student_ids]
        sample_carreers = full_carreers.copy(deep=True)
        #We never use full carreers again so we can do a shallow copy
        drop_courses = full_carreers.copy()
        for i, student in full_carreers.iterrows():
            drop_course = (student.dropna().sample(int(student.dropna().count() * (1-proportion_classes_taken))))
            drop_courses.loc[i, ~drop_courses.columns.isin(drop_course.index)] = np.nan # type: ignore
            sample_carreers.loc[i,drop_course.index] = np.nan # type: ignore
        #The temp matrix is used to store our student-course matrix but with our selected students sample carreers
        temp_matrix = self.student_course_matrix.copy(deep=True)
        temp_matrix.loc[sample_carreers.index.tolist()] = sample_carreers
        #Make a dataframe to load our final reccomendations into
        reccomendations = pd.DataFrame(index=student_ids, columns=temp_matrix.columns)
        #Loop over each selected student and fill the respective row in our reccomendation dataframe
        for student_id in student_ids:
            #Calculate cosign similarity for all students on only courses the selected student has taken, see single reccomdation for reasoning
            student_similarity = pd.DataFrame(cosine_similarity(temp_matrix[sample_carreers.loc[student_id].dropna().index].fillna(0)), index=temp_matrix.index, columns=temp_matrix.index)
            student_similarity.drop(student_ids, inplace=True)
            simiar_students = student_similarity[student_id].sort_values(ascending=False)[:num_similar_students] # type: ignore
            #Drop taken courses from our reccomendations
            taken_courses = temp_matrix.loc[student_id].dropna()
            reccomended_courses = temp_matrix.loc[simiar_students.index].dropna(axis=1, how='all')
            reccomended_courses.drop(taken_courses.index, axis=1, inplace=True, errors='ignore')
            #Generate a temp dict to score course id -> predicted score for our classes
            #Same as single reccomendation
            reccomendation = {}
            for course in reccomended_courses.columns:
                course_scores = reccomended_courses[course]
                course_score, course_count = 0,0
                for student in simiar_students.index:
                    if pd.isna(course_scores[student]): continue
                    course_score += course_scores[student] * simiar_students[student]
                    course_count += simiar_students[student]
                reccomendation[course] = course_score/course_count
            reccomendations.loc[student_id, reccomendation.keys()] = reccomendation# type: ignore
        return (sample_carreers, drop_courses, reccomendations)