printf("load eigenVec.. \n");
load eigenVec;
eigenVectors = eigenVec;
% size(eigenVectors) %6714x6714

printf("load eigenVal.. \n");
load eigenVal;
eigenValues = eigenVal;
% size(eigenValues) %6714x1

printf("sorting.. \n");
[eigenValues_sort I] = sort(eigenValues, 'descend');
% eigenValues_sort
% save dd.dat eigenValues_sort
eigenVectors_sort = eigenVectors(:,I);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% printf("computing scores.. \n ");
% sc = cumsum(eigenValues_sort)./sum(eigenValues_sort);
% K = 0;
% for i = 1:size(sc)(1)
%     if sc(i) >= 0.95 %0.95=>K=1703
%         K = i
%         break
%     end
% end
K = 2000;

printf("find transMatrix.. \n");
transMatrix = eigenVectors_sort(:,1:K);
size(transMatrix) %6714xK

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
printf("load train_mtx.csv.. \n");
load train_mtx.csv;
dataMatrix = train_mtx;
size(dataMatrix) %39774x6714
% printf("load test_mtx.csv.. \n");
% load test_mtx.csv;
% dataMatrix = test_mtx;
% size(dataMatrix) %9944x6714

printf("find newMatrix.. \n");
newMatrix = dataMatrix * transMatrix;
for i = 1:K
    Max_i = max(newMatrix(:,i));
    Min_i = min(newMatrix(:,i));
    newMatrix(:,i) = ( newMatrix(:,i) - Min_i ) / ( Max_i - Min_i );
end
size(newMatrix) %train:39774xK; test:9944xK

printf("write file.. \n");
dlmwrite('pca_mtx.csv', newMatrix, 'precision', 9);

