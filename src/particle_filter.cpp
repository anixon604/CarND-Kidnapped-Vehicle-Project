/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	num_particles = 1000;

	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	default_random_engine gen;

	for(int i = 0; i < num_particles; i++) {
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;

		particles.push_back(p);
		weights.push_back(p.weight);
	}



	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// xf = x0 + v/yaw_rate [sin(theta + yaw_rate(dt)) - sin(theta)]
	// yf = y0 + v/yaw_rate [cos(theta) - cos(theta + yaw_rate(dt))]
	// heading = theta + yaw_rate(dt)

	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];

	double v_div_y = velocity / yaw_rate;
	double y_mul_t = yaw_rate * delta_t;

	default_random_engine gen;

	for(Particle& p : particles) {
		// add measurements
		p.x = p.x + v_div_y * (sin(p.theta + y_mul_t) - sin(p.theta));
		p.y = p.y + v_div_y * (cos(p.theta) - cos(p.theta + y_mul_t));
		p.theta = p.theta + y_mul_t;

		normal_distribution<double> dist_x(p.x, std_x);
		normal_distribution<double> dist_y(p.y, std_y);
		normal_distribution<double> dist_theta(p.theta, std_theta);

		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
	}

}

/**
	* helper function that transforms the observations of MAP coordinates (particles and landmarks)
	* to VEHICLE (observations)
	* - Map coordinates will be translated by particle position
	*	- Map coordinates will be rotated to partical heading angle
	* @ param observations Vector of landmark observations
  */
std::vector<LandmarkObs> transformCoords(double sensor_range, Particle p, Map map_landmarks) {

	std::vector<LandmarkObs> predicted;

	double partX = p.x;
	double partY = p.y;
	double partHeading = p.theta;
	double landX, landY;

	// if inverting from a map coord to a car relative. Then translation
	// must be BEFORE rotation.
	// (xfinal - carx)*cos(-heading) - (yfinal - cary)*sin(-heading)
	// (xfinal - carx)*sin(-heading) + (yfinal - cary)*cos(-heading)
	for(int i = 0; i < map_landmarks.landmark_list.size(); i++) {
		Map::single_landmark_s landmark = map_landmarks.landmark_list[i];
		landX = landmark.x_f;
		landY = landmark.y_f;

		LandmarkObs l;
		l.id = landmark.id_i;
		l.x = (landX-partX)*cos(-partHeading)-(landY-partY)*sin(-partHeading);
		l.y = (landX-partX)*sin(-partHeading)+(landY-partY)*cos(-partHeading);

		// TODO: implement sensor_range filter

		predicted.push_back(l);
	}


	return predicted;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	// sort the observations by ID
	struct {
		bool operator()(LandmarkObs a, LandmarkObs b) {
			return a.id < b.id;
		}
	} LandmarkCompare;
	sort(observations.begin(), observations.end(), LandmarkCompare);

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// vector<LandmarkObs> predicted - a simulated observation of each landmark relative to the particle.
	vector<LandmarkObs> predicted;

	for(int i = 0; i < num_particles; i++) {
			predicted = transformCoords(sensor_range, particles[i], map_landmarks);
			dataAssociation(predicted, observations);

			// BEGIN calculate Multivariate_normal_distribution
			/* Covariance Matrix E:
					| variance_x  0 |
					| 0  variance_y |

				 Inverse of Covariance Matrix E:
				 1/[variance_x*variance_y] * | variance_y   0 |
				 									 					 | 0   variance_x |
			*/
			double var_x = std_landmark[0]*std_landmark[0];
			double var_y = std_landmark[1]*std_landmark[1];
			double invCovX = 1/var_x;
			double invCovY = 1/var_y;
			double delta_x, delta_y, ximu2, yimu2;

			double weight = 0;

			// note: predicted is already sorted by id.
			// 			 since it was generated from map_landmarks which was ordered
			//			 at creation.
			// observations were sorted by id in dataAssociation() using custom
			// comparitor.
			// - Will only process predicted landmarks that were matched to observations
			for(int i = 0; i < observations.size(); i++) {
				int landmark_id = observations[i].id;
				delta_x = predicted[landmark_id-1].x - observations[i].x;
				delta_y = predicted[landmark_id-1].y - observations[i].y;
				ximu2 = delta_x*delta_x;
				yimu2 = delta_y*delta_y;

				double numerator = exp((-1/2)*(ximu2*invCovX + yimu2*invCovY));
				// sqrt[|2PI*E|] = sqrt[determinant of 2PI * E]
				double denominator = sqrt((1/(pow(2*M_PI, 2)))*invCovX*invCovY);

				//END calculate Multivariate_normal_distribution

				weight *= numerator/denominator;
			}

			particles[i].weight = weight; // assign new weight to particle.

	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
