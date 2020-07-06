(function () {
  'use strict';
  angular.module('ThoughtAnalyzerApp', [])
  .controller('ThoughtAnalyzerController', ['$scope', '$log', '$http',
      function($scope, $log, $http, $timeout) {
        $scope.getResults = function() {

            $log.log("test");

            // get the URL from the input
            var userInput = $scope.text;
               $log.log("userInput");
            // fire the API request
            $http.post('/sentiment', {"text": userInput}).
              success(function(results) {
                $log.log(results);
                getSentiment(results);
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
                  $timeout.cancel(timeout);
                  $scope.results = data;
                  return false;
                }
                // continue to call the poller() function every 2 seconds
                // until the timeout is cancelled
                timeout = $timeout(poller, 2000);
              });
          };
          poller();
        };
  }]);
});