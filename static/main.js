(function () {

  'use strict';

  angular.module('ThoughtAnalyzerApp', [])

  .controller('ThoughtAnalyzerController', ['$scope', '$log', '$http',
      function($scope, $log, $http, $timeout) {

        $scope.getResults = function() {

            $log.log("test");

            // get the URL from the input
            var userInput = $scope.text;
            $log.log(userInput);
            // fire the API request
            $http.post('/sentiment', {"text": userInput}).
              success(function(results) {
                $log.log(results);
              }).
              error(function(error) {
                $log.log(error);
              })
        };
      }
      ]);

}());
