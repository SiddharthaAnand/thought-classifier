(function () {

  'use strict';

  angular.module('ThoughtAnalyzerApp', [])

  .controller('ThoughtAnalyzerController', ['$scope', '$log', '$http', '$timeout',
      function($scope, $log, $http, $timeout) {
        $scope.submitButtonText = 'Submit';
        $scope.loading = false;
        $scope.getResults = function() {

            $log.log("test");

            // get the URL from the input
            var userInput = $scope.text;
            $log.log(userInput);
            // fire the API request
            $http.post('/sentiment', {"text": userInput}).
              success(function(jobId) {
                $log.log(jobId);
                getSentiment(jobId);
                $scope.results = null;
                $scope.submitButtonText = 'Loading...';
                $scope.loading = true;
              }).
              error(function(error) {
                $log.log(error);
              });
        };

      function getSentiment(jobID) {

      var timeout = "";

      var poller = function() {
        // fire another request
        $http.get('/sentiment/'+jobID).
          success(function(data, status, headers, config) {
            if(status === 202) {
              $log.log(data, status);
            } else if (status === 200){
              $log.log(data);
              $scope.results = data;
              $timeout.cancel(timeout);
              $scope.submitButtonText = 'Submit';
              $scope.loading = false;
              return false;
            }
            // continue to call the poller() function every 2 seconds
            // until the timeout is cancelled
            timeout = $timeout(poller, 2000);
          });
      };
      poller();
    }
      }
      ]);

}());
