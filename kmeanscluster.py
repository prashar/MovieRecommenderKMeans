from sklearn.cluster import KMeans
from numpy import *
from scipy.spatial.distance import cdist
from matplotlib.pyplot import *
from matplotlib.cm import *
class kmeanscluster:

    def __init__(self):
        self.clusters = 5
        self.initCentroids = 'random'
        self.iters = 10
        self.times = 1
        self.features = 2
        self.samples = 1
    
    # --------------------------------------------------------------------------------
    # Test Code ... 
    # --------------------------------------------------------------------------------
    def read_data(self):
        X = loadtxt("hw4-data/mnist2500_X.txt")
        Y = loadtxt("hw4-data/mnist2500_labels.txt")
        return X,Y

    def trivial(self):
        arr = array([[1,1.0,1.0],[2,1.5,2.0],[3,3.0,4.0],[4,5.0,7.0],[5,3.5,5.0],[6,4.5,5.0],[7,3.5,4.5]])
        kmeans = KMeans(init='random',n_clusters=2,n_init=1,verbose=1,max_iter=40)
        kmeans.fit(arr[:,1:])
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        print cluster_centers
        print labels
    
    def sklearn_kmeans(self,X,num_clusters,n_init,max_iters):
        centers = X[0:20,:]
        kmeans = KMeans(init=centers,n_clusters=num_clusters,n_init=n_init,verbose=1,max_iter=max_iters)
        kmeans.fit(X)
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        print shape(labels)
        for i in range(20):
            print shape(where(labels == i)[0])

    # --------------------------------------------------------------------------------
    # Group Sum of Squares
    # --------------------------------------------------------------------------------
    def GroupSumSquares(self,data,cluster_size,labels,centers):
        #print 'label_shape'
        #print shape(labels)
        #print shape(data)

        # Maintain counter per clusterID
        dwCountPerCluster = zeros((1,cluster_size))
        fSSPerCluster = zeros((1,cluster_size))

        for i in range(cluster_size):
            dwCountPerCluster[0,i] = shape(where(labels == i)[0])[0]
            bMask = (labels == i)
            #print 'bmask', shape(bMask)
            cluster_center = centers[i].reshape(1,self.features)
            cluster_points = data[bMask.flatten()]
            #print cluster_points
            #print shape(cluster_points)
            #print shape(cluster_center)

            # We are not taking the sqrt, since the SS(k) term is squared.
            ss = sum(sum((cluster_points - cluster_center)**2,axis=1))
            fSSPerCluster[0,i] = ss

        #print dwCountPerCluster
        #print fSSPerCluster
        print "Total SS for k =  ", cluster_size, ' ', sum(fSSPerCluster.flatten())
        return dwCountPerCluster,fSSPerCluster

    # --------------------------------------------------------------------------------
    # Mistake Rate
    # --------------------------------------------------------------------------------
    def CalculateMistakeRate(self,data,cluster_size,labels,centers,truth_labels):
        #print shape(labels)
        #print type(labels)
        #print shape(truth_labels)
        mistake_registrar = zeros(cluster_size)
        for cluster_id in range(cluster_size):
            cluster_mask = (labels == cluster_id)
            cluster_samples = truth_labels[cluster_mask.flatten()]
            #print shape(cluster_samples)
            bin_count = bincount((cluster_samples.astype(int)).flatten())
            nonzero_binindices = nonzero(bin_count)[0]
            max_vote_idx = argmax(bin_count)
            max_vote_value = bin_count[max_vote_idx]
            cluster_mistakes = sum(bin_count) - max_vote_value
            #print bin_count
            print "Max Vote: " , max_vote_idx, ' ' ,  max_vote_value
            mistake_registrar[cluster_id] = cluster_mistakes
            print "Mistakes Made in this cluster : ", cluster_mistakes
            #vstack((nonzero_binindices,bin_count[nonzero_binindices]))
            #exit()
        print "Total Mistakes"
        return mistake_registrar

    # --------------------------------------------------------------------------------
    # Show Binary Images
    # --------------------------------------------------------------------------------

    def ShowBinaryImages(self,data,labels,clusterSmallError,clusterLargeError):
        for cluster_id in [clusterSmallError,clusterLargeError]:
            cluster_mask = (labels == cluster_id)
            cluster_samples = data[cluster_mask.flatten()]
            range_to_choose_from = range(shape(cluster_samples)[0])
            random.shuffle(range_to_choose_from)
            dwThreeSamples = range_to_choose_from[:3]
            print "Indices for the three random samples we'll display " , dwThreeSamples
            for idx in dwThreeSamples:
                figure(1)
                imshow(reshape(cluster_samples[idx,:],(28,28)), cmap = cm.binary)
                show()

    # --------------------------------------------------------------------------------
    # Q 2.5.1
    # --------------------------------------------------------------------------------

    def CalculateGSSAndMR(self,X,Y,max_iters,num_runtimes):
        dwClusters = array([19])
        for cluster in dwClusters:
            centroids = X[0:cluster,:]
            centers,labels,iters = self.fit_cluster(cluster,X,centroids,num_runtimes,max_iters)
            dwClusterCount, fGss = self.GroupSumSquares(X,cluster,labels,centers)
            print '==> Total rate for cluster size ', cluster, ' :', sum(fGss)
            dwMistakesPerCluster = self.CalculateMistakeRate(X,cluster,labels,centers,Y)
            print '==> Mistakes & MR for cluster ', cluster, ' : '
            print sum(dwMistakesPerCluster), ' : ', (sum(dwMistakesPerCluster)/self.samples)
            print '==> Mistakes for cluster      ', cluster, ' : '
            print dwMistakesPerCluster
            print '==> Cluster Count             ', cluster, ' : '
            print dwClusterCount
            print '==> Mistake Rate for cluster  ', cluster, ' : '
            fMistakeRatePerCluster = dwMistakesPerCluster/dwClusterCount
            dwBest = argmin(fMistakeRatePerCluster.flatten())
            dwWorst = argmax(fMistakeRatePerCluster.flatten())
            if(cluster == 20):
                plot(range(20),fMistakeRatePerCluster.flatten())
                title("Mistake Rates for each cluster for k=20")
                ylabel("Mistake Rate")
                xlabel("k")
                grid()
                show()
            print fMistakeRatePerCluster.flatten()
            print "Worst Mistake Rate for cluster ", dwWorst, " : " , fMistakeRatePerCluster.flatten()[dwWorst]
            print "Best Mistake Rate for cluster ", dwBest, " : " , fMistakeRatePerCluster.flatten()[dwBest]
            if(cluster == 20):
                self.ShowBinaryImages(X,labels,dwBest,dwWorst)

    # --------------------------------------------------------------------------------
    # Q 2.5.6/7
    # --------------------------------------------------------------------------------
    def CalculateRandGSS(self,X,Y,max_iters,num_runtimes):
        mistake_rate = zeros((20,1))
        gss_rate = zeros((20,1))
        idx_main=0

        dwClusters = range(1,21)
        for cluster in dwClusters:
            total_mistakes = 0
            total_gss = 0
            for i in range(5):
                centers,labels,iters = self.fit_cluster(cluster,X,'rand',num_runtimes,max_iters)
                dwClusterCount, fGss = self.GroupSumSquares(X,cluster,labels,centers)
                print '==> Total GSS for cluster size ', cluster, ' :', sum(fGss)
                dwMistakesPerCluster = self.CalculateMistakeRate(X,cluster,labels,centers,Y)
                print '==> Mistakes & MR for cluster ', cluster, ' : '
                print sum(dwMistakesPerCluster), ' : ', (sum(dwMistakesPerCluster)/self.samples)
                print '==> Mistakes for cluster      ', cluster, ' : '
                print dwMistakesPerCluster
                print '==> Cluster Count             ', cluster, ' : '
                print dwClusterCount
                print '==> Mistake Rate for cluster  ', cluster, ' : '
                fMistakeRatePerCluster = dwMistakesPerCluster/dwClusterCount
                print fMistakeRatePerCluster
                # Keep adding to the totals
                total_mistakes += sum(dwMistakesPerCluster)
                total_gss += sum(fGss)

            mistake_rate[idx_main,0] = (total_mistakes/5.)/self.samples
            gss_rate[idx_main,0] = total_gss/5.
            idx_main += 1

        # Plotting Mistake Rate
        plot(range(1,21),mistake_rate.flatten())
        title("Mistake Rates / k")
        ylabel("Mistake Rate")
        xlabel("k")
        grid()
        show()

        # Plotting GSS
        plot(range(1,21),gss_rate.flatten())
        title("Group Sum of Squares")
        ylabel("Within Group Sum of Squares")
        xlabel("k")
        grid()
        show()

    # --------------------------------------------------------------------------------
    # MY ALGORITHM FOR K_MEANS
    # --------------------------------------------------------------------------------

    def fit_cluster(self,num_clusters,data,initCentroids,numTimes,maxIters):

        print("=======>Running for k = ", num_clusters)

        # Update parameters based on data ..
        self.features = shape(data)[1]
        self.samples = shape(data)[0]
        self.iters = maxIters
        self.clusters = num_clusters

        curr_iters = 0

        # We will maintain an array of size(clusters,features)
        if(type(initCentroids).__name__ == 'str'):
            print "Random Initialization of Centroids"
            indices = arange(self.samples)
            random.shuffle(indices)
            print "Indices chosen: ", indices
            indices = indices[:self.clusters]
            self.initCentroids = data[indices.flatten()]
        else:
            print "Initial Centroids Provided"
            self.initCentroids = initCentroids.copy()

        #print shape(self.initCentroids)

        # This will maintain old values of the distances
        prev_centers = zeros((self.clusters,self.features))

        # We'll maintain a running sum of the difference between the old value
        # and the new value of centroids, and terminate if we don't

        while((self.iters > curr_iters) and (sum(sqrt(sum((self.initCentroids-prev_centers)**2,axis=1))) != 0)):
            # Save the prev vector
            prev_centers = copy(self.initCentroids)

            # Get euclidean dist to centroid for Cluster 0 for all samples.
            dist_clusterX = sum((data - self.initCentroids[0,:])**2,axis=1).reshape(1,self.samples)
            #print shape(dist_clusterX)
            # Need to create a cluster x sample array all the distances are for
            # a sample are in one column, so we need to append each
            # dist_clusterX weight vector with the previous

            # Must increment before
            curr_iters += 1
            for i in range(1,self.clusters):
                dist_cluster_i = sum((data-self.initCentroids[i,:])**2,axis=1).reshape(1,self.samples)
                #print shape(dist_cluster_i)
                dist_clusterX = append(dist_clusterX, dist_cluster_i, axis=0)

            #print shape(dist_clusterX.T)
            #print dist_clusterX

            # This will be a clusters x samples size array
            dist_all_clusters = argmin(dist_clusterX,axis=0)
            #print dist_all_clusters

            #print shape(dist_all_clusters)

            # Now we need to somehow sort all the vectors together
            sample_separator = ones((1,self.samples)) * dist_all_clusters
            #print sample_separator
            #print "------"
            #print shape(sample_separator.T)

            for cluster in range(self.clusters):
                cluster_x = where(sample_separator.T == cluster, 1, 0)
                if(sum(cluster_x) > 0):
                    cluster_x_size = sum(cluster_x)
                    cluster_x_totalsum = sum(cluster_x * data, axis=0)
                    self.initCentroids[cluster,:] = cluster_x_totalsum/cluster_x_size
            #print sum(self.initCentroids == prev_centers)

        print "====> Completed in ", curr_iters, " iterations .. "
        '''
        for i in range(self.clusters):
            print "-- ", i
	    print shape(where(sample_separator.T == i)[0])
        '''
        return self.initCentroids,sample_separator.T,curr_iters


def main():
    cluster = kmeanscluster()
    #cluster.trivial()
    X,Y = cluster.read_data()
    #cluster.sklearn_kmeans(X,20,1,40)

    #centroids = X[0:20,:]
    #centers,labels,iters = cluster.fit_cluster(20,X,centroids,1,40)

    cluster.CalculateGSSAndMR(X,Y,40,1)
    #cluster.CalculateRandGSS(X,Y,40,5)


    #print getattr(cluster,'num_clusters')
    #print hasattr(cluster,'num_clusters')


if __name__=='__main__':
    main()
