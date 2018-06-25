// This file provides fast analysis functions written in c++. This file is needed to circumvent some python limitations where
// no sufficient pythonic solution is available.
#pragma once

#include <iostream>
#include <string>
#include <ctime>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <algorithm>
#include <sstream>

#include "defines.h"

bool _debug = false;
bool _info = false;

// Counts from the event number column of the cluster table how often a cluster occurs in every event
unsigned int getNclusterInEvents(int64_t*& rEventNumber, const unsigned int& rSize, int64_t*& rResultEventNumber, unsigned int*& rResultCount)
{
	unsigned int tResultIndex = 0;
	unsigned int tLastIndex = 0;
	int64_t tLastValue = 0;
	for (unsigned int i = 0; i < rSize; ++i) {  // loop over all events can count the consecutive equal event numbers
		if (i == 0)
			tLastValue = rEventNumber[i];
		else if (tLastValue != rEventNumber[i]) {
			rResultCount[tResultIndex] = i - tLastIndex;
			rResultEventNumber[tResultIndex] = tLastValue;
			tLastValue = rEventNumber[i];
			tLastIndex = i;
			tResultIndex++;
		}
	}
	// add last event
	rResultCount[tResultIndex] = rSize - tLastIndex;
	rResultEventNumber[tResultIndex] = tLastValue;
	return tResultIndex + 1;
}

// Takes two event arrays and calculates an intersection array of event numbers occurring in both arrays
unsigned int getEventsInBothArrays(int64_t*& rEventArrayOne, const unsigned int& rSizeArrayOne, int64_t*& rEventArrayTwo, const unsigned int& rSizeArrayTwo, int64_t*& rEventArrayIntersection)
{
	int64_t tActualEventNumber = -1;
	unsigned int tActualIndex = 0;
	unsigned int tActualResultIndex = 0;
	for (unsigned int i = 0; i < rSizeArrayOne; ++i) {  // loop over all event numbers in first array
		if (rEventArrayOne[i] == tActualEventNumber)  // omit the same event number occuring again
			continue;
		tActualEventNumber = rEventArrayOne[i];
		for (unsigned int j = tActualIndex; j < rSizeArrayTwo; ++j) {
			if (rEventArrayTwo[j] >= tActualEventNumber) {
				tActualIndex = j;
				break;
			}
		}
		if (rEventArrayTwo[tActualIndex] == tActualEventNumber) {
			rEventArrayIntersection[tActualResultIndex] = tActualEventNumber;
			tActualResultIndex++;
		}
	}
	return tActualResultIndex++;
}

// Takes two event number arrays and returns a event number array with the maximum occurrence of each event number in array one and two
unsigned int getMaxEventsInBothArrays(int64_t*& rEventArrayOne, const unsigned int& rSizeArrayOne, int64_t*& rEventArrayTwo, const unsigned int& rSizeArrayTwo, int64_t*& result, const unsigned int& rSizeArrayResult)
{
	int64_t tFirstActualEventNumber = rEventArrayOne[0];
	int64_t tSecondActualEventNumber = rEventArrayTwo[0];
	int64_t tFirstLastEventNumber = rEventArrayOne[rSizeArrayOne - 1];
	int64_t tSecondLastEventNumber = rEventArrayTwo[rSizeArrayTwo - 1];
	unsigned int i = 0;
	unsigned int j = 0;
	unsigned int tActualResultIndex = 0;
	unsigned int tFirstActualOccurrence = 0;
	unsigned int tSecondActualOccurrence = 0;

	bool first_finished = false;
	bool second_finished = false;

//	std::cout<<"tFirstActualEventNumber "<<tFirstActualEventNumber<<std::endl;
//	std::cout<<"tSecondActualEventNumber "<<tSecondActualEventNumber<<std::endl;
//	std::cout<<"tFirstLastEventNumber "<<tFirstLastEventNumber<<std::endl;
//	std::cout<<"tSecondLastEventNumber "<<tSecondLastEventNumber<<std::endl;
//	std::cout<<"rSizeArrayOne "<<rSizeArrayOne<<std::endl;
//	std::cout<<"rSizeArrayTwo "<<rSizeArrayTwo<<std::endl;
//	std::cout<<"rSizeArrayResult "<<rSizeArrayResult<<std::endl;

	while (!(first_finished && second_finished)) {
		if ((tFirstActualEventNumber <= tSecondActualEventNumber) || second_finished) {
			unsigned int ii;
			for (ii = i; ii < rSizeArrayOne; ++ii) {
				if (rEventArrayOne[ii] == tFirstActualEventNumber)
					tFirstActualOccurrence++;
				else
					break;
			}
			i = ii;
		}

		if ((tSecondActualEventNumber <= tFirstActualEventNumber) || first_finished) {
			unsigned int jj;
			for (jj = j; jj < rSizeArrayTwo; ++jj) {
				if (rEventArrayTwo[jj] == tSecondActualEventNumber)
					tSecondActualOccurrence++;
				else
					break;
			}
			j = jj;
		}

//		std::cout<<"tFirstActualEventNumber "<<tFirstActualEventNumber<<" "<<tFirstActualOccurrence<<" "<<first_finished<<std::endl;
//		std::cout<<"tSecondActualEventNumber "<<tSecondActualEventNumber<<" "<<tSecondActualOccurrence<<" "<<second_finished<<std::endl;

		if (tFirstActualEventNumber == tSecondActualEventNumber) {
//			std::cout<<"==, add "<<std::max(tFirstActualOccurrence, tSecondActualOccurrence)<<" x "<<tFirstActualEventNumber<<std::endl;
			if (tFirstActualEventNumber == tFirstLastEventNumber)
				first_finished = true;
			if (tSecondActualEventNumber == tSecondLastEventNumber)
				second_finished = true;
			for (unsigned int k = 0; k < std::max(tFirstActualOccurrence, tSecondActualOccurrence); ++k) {
				if (tActualResultIndex < rSizeArrayResult)
					result[tActualResultIndex++] = tFirstActualEventNumber;
				else
					throw std::out_of_range("The result histogram is too small. Increase size.");
			}
		}
		else if ((!first_finished && tFirstActualEventNumber < tSecondActualEventNumber) || second_finished) {
//			std::cout<<"==, add "<<tFirstActualOccurrence<<" x "<<tFirstActualEventNumber<<std::endl;
			if (tFirstActualEventNumber == tFirstLastEventNumber)
				first_finished = true;
			for (unsigned int k = 0; k < tFirstActualOccurrence; ++k) {
				if (tActualResultIndex < rSizeArrayResult)
					result[tActualResultIndex++] = tFirstActualEventNumber;
				else
					throw std::out_of_range("The result histogram is too small. Increase size.");
			}
		}
		else if ((!second_finished && tSecondActualEventNumber < tFirstActualEventNumber) || first_finished) {
//			std::cout<<"==, add "<<tSecondActualOccurrence<<" x "<<tSecondActualEventNumber<<std::endl;
			if (tSecondActualEventNumber == tSecondLastEventNumber)
				second_finished = true;
			for (unsigned int k = 0; k < tSecondActualOccurrence; ++k) {
				if (tActualResultIndex < rSizeArrayResult)
					result[tActualResultIndex++] = tSecondActualEventNumber;
				else
					throw std::out_of_range("The result histogram is too small. Increase size.");
			}
		}

		if (i < rSizeArrayOne)
			tFirstActualEventNumber = rEventArrayOne[i];
		if (j < rSizeArrayTwo)
			tSecondActualEventNumber = rEventArrayTwo[j];
		tFirstActualOccurrence = 0;
		tSecondActualOccurrence = 0;
	}

	return tActualResultIndex;
}

// Does the same as np.in1d but uses the fact that the arrays are sorted
void in1d_sorted(int64_t*& rEventArrayOne, const unsigned int& rSizeArrayOne, int64_t*& rEventArrayTwo, const unsigned int& rSizeArrayTwo, uint8_t*& rSelection)
{
	rSelection[0] = true;
	int64_t tActualEventNumber = -1;
	unsigned int tActualIndex = 0;
	for (unsigned int i = 0; i < rSizeArrayOne; ++i) {  // loop over all event numbers in first array
		tActualEventNumber = rEventArrayOne[i];
		for (unsigned int j = tActualIndex; j < rSizeArrayTwo; ++j) {
			if (rEventArrayTwo[j] >= tActualEventNumber) {
				tActualIndex = j;
				break;
			}
		}
		if (rEventArrayTwo[tActualIndex] == tActualEventNumber)
			rSelection[i] = 1;
		else
			rSelection[i] = 0;
	}
}

// Fast 1d index histogramming (bin size = 1, values starting from 0)
void histogram_1d(int*& x, const unsigned int& rSize, const unsigned int& rNbinsX, uint32_t*& rResult)
{
	for (unsigned int i = 0; i < rSize; ++i) {
		if (x[i] >= rNbinsX){
			std::stringstream errorString;
			errorString << "The histogram index x=" << x[i] << " is out of range.";
			throw std::out_of_range(errorString.str());
		}
		if (rResult[x[i]] < 4294967295)
			++rResult[x[i]];
		else
			throw std::out_of_range("The histogram has more than 4294967295 entries per bin. This is not supported.");
	}
}

// Fast 2d index histogramming (bin size = 1, values starting from 0)
void histogram_2d(int*& x, int*& y, const unsigned int& rSize, const unsigned int& rNbinsX, const unsigned int& rNbinsY, uint32_t*& rResult)
{
	for (unsigned int i = 0; i < rSize; ++i) {
		if (x[i] >= rNbinsX || y[i] >= rNbinsY){
			std::stringstream errorString;
			errorString << "The histogram indices (x/y)=(" << x[i] << "/" << y[i] << ") are out of range.";
			throw std::out_of_range(errorString.str());
		}
		if (rResult[x[i] * rNbinsY + y[i]] < 4294967295)
			++rResult[x[i] * rNbinsY + y[i]];
		else
			throw std::out_of_range("The histogram has more than 4294967295 entries per bin. This is not supported.");
	}
}

// Fast 3d index histogramming (bin size = 1, values starting from 0)
void histogram_3d(int*& x, int*& y, int*& z, const unsigned int& rSize, const unsigned int& rNbinsX, const unsigned int& rNbinsY, const unsigned int& rNbinsZ, uint16_t*& rResult)
{
	for (unsigned int i = 0; i < rSize; ++i) {
		if (x[i] >= rNbinsX || y[i] >= rNbinsY || z[i] >= rNbinsZ) {
			std::stringstream errorString;
			errorString << "The histogram indices (x/y/z)=(" << x[i] << "/" << y[i] << "/" << z[i] << ") are out of range.";
			throw std::out_of_range(errorString.str());
		}
		if (rResult[x[i] * rNbinsY * rNbinsZ + y[i] * rNbinsZ + z[i]] < 65535)
			++rResult[x[i] * rNbinsY * rNbinsZ + y[i] * rNbinsZ + z[i]];
		else
			throw std::out_of_range("The histogram has more than 65535 entries per bin. This is not supported.");
	}
}

// loop over the refHit, Hit arrays and compare the hits of same event number. If they are similar (within an error) correlation is assumed. If more than nBadEvents are not correlated, broken correlation is assumed.
// True/False is returned for correlated/not correlated data. The iRefHit index is the index of the first not correlated hit.
bool _checkForNoCorrelation(unsigned int& iRefHit, unsigned int& iHit, const int64_t*& rEventArray, const double*& rRefCol, double*& rCol, const double*& rRefRow, double*& rRow, uint8_t*& rCorrelated, const unsigned int& nHits, const double& rError, const unsigned int& nBadEvents)
{
	int64_t tRefEventNumber = rEventArray[iRefHit];  // last read reference hit event number
	int64_t tEventNumberOffset = rEventArray[iHit] - rEventArray[iRefHit];  // event number offset between reference hit and hit
	unsigned int tBadEvents = 0;  // consecutive not correlated events
	unsigned int tHitIndex = iRefHit;  // actual first not correlated hit index
	bool tIsCorrelated = false;  // flag for the actual event
	unsigned int tNVirtual = 0;  // number of pure virtual events (only have virtual hits)
	unsigned int tNrefHits = 0; // number of reference hits (including virtual) of actual event
	unsigned int tNHits = 0; // number of hits (including virtual) of actual event

	for (; iRefHit < nHits && iHit < nHits; ++iRefHit, ++iHit) {
		while (iRefHit < nHits && iHit < nHits && (rEventArray[iRefHit] + tEventNumberOffset) != rEventArray[iHit]) {  // reference hit and hit array are not in sync --> correct
			while (iRefHit < nHits && ((rEventArray[iRefHit] + tEventNumberOffset) < rEventArray[iHit])) {  // hit array is at next event, catch up with reference hit array
				iRefHit++;
				tNrefHits++;
			}
			while (iHit < nHits && ((rEventArray[iRefHit] + tEventNumberOffset) > rEventArray[iHit])) {  // reference hit array is at next event, catch up with hit array
				iHit++;
				tNHits++;
			}
		}

		if (iRefHit == nHits || iHit == nHits)  // one array is at the end, abort copying
			break;

		if (tRefEventNumber != rEventArray[iRefHit]) {  // new event trigger
			if (!tIsCorrelated) {
				if (tBadEvents == 0) {
					tHitIndex = iHit;
					for (tHitIndex; tHitIndex > 0; --tHitIndex) {  // the actual first not correlated hit is the first hit of the last event
						if (rEventArray[tHitIndex] < tRefEventNumber) {
							tHitIndex++;
							break;
						}
					}
					if (rEventArray[tHitIndex] < tRefEventNumber)
						tHitIndex++;
				}
				if (tNVirtual == tNHits && tNVirtual == tNrefHits)  // if there are only virtual hits one cannot judge the correlation, do not increase bad event counter
					tBadEvents++;
			}
			else
				tBadEvents = 0;

			tRefEventNumber = rEventArray[iRefHit];
			tIsCorrelated = false;
			tNVirtual = 0;
		}

		if (tBadEvents >= nBadEvents) {  // a correlation is defined as broken if more than nBadEvents consecutive, not correlated events exist
			iRefHit = tHitIndex;  // set reference hit / hit to first not correlated hit
			iHit = tHitIndex;
			return true;
		}

		if (rRefCol[iRefHit] != 0 && rCol[iHit] != 0 && rRefRow[iRefHit] != 0 && rRow[iHit] != 0 && std::fabs(rRefCol[iRefHit] - rCol[iHit]) < rError && std::fabs(rRefRow[iRefHit] - rRow[iHit]) < rError)  // check for correlation of real hits
			tIsCorrelated = true;
		if ((rRefCol[iRefHit] == 0 && rCol[iHit] == 0 && rRefRow[iRefHit] == 0 && rRow[iHit] == 0) || rCorrelated[iHit] == 0)  // if virtual hits occur in both devices correlation is likely
			tNVirtual++;
		if (_debug)
			std::cout << "\n" << iRefHit << "\t" << iHit << "\t" << rEventArray[iRefHit] << "\t" << rEventArray[iHit] << "\t" << rRefRow[iRefHit] << " / " << rRow[iHit] << "\t" << (int) rCorrelated[iHit] << "\t" << tNVirtual << "\t" << tIsCorrelated << "\t" << tBadEvents << "\n";
	}
	// Correct out of boundary indices
	if (iRefHit == nHits)
		iRefHit--;
	if (iHit == nHits)
		iHit--;
	return false;
}

// loop over the refHit, Hit arrays and compare the hits of same event number. If they are similar (within an error) correlation is assumed. If more than nBadEvents are not correlated, broken correlation is assumed.
// True/False is returned for correlated/not correlated data. The iRefHit index is the index of the first not correlated hit.
bool _checkForCorrelation(unsigned int iRefHit, unsigned int iHit, const int64_t*& rEventArray, const double*& rRefCol, double*& rCol, const double*& rRefRow, double*& rRow, uint8_t*& rCorrelated, const unsigned int& nHits, const unsigned int& rStopRefHitIndex, const unsigned int& rStopHitIndex, const double& rError, const unsigned int nGoodEvents, bool print = false)
{
	int64_t tRefEventNumber = rEventArray[iRefHit];  // last read reference hit event number
	int64_t tEventNumberOffset = rEventArray[iHit] - rEventArray[iRefHit];  // event number offset between reference hit and hit
	unsigned int tNgoodEvents = 0;  // consecutive correlated events
	unsigned int tHitIndex = iRefHit;  // actual first not correlated hit index
	bool tIsCorrelated = false;  // flag for the actual event
	unsigned int tNVirtual = 0;  // number of pure virtual events (only have virtual hits)
	unsigned int tNrefHits = 0; // number of reference hits (including virtual) of actual event
	unsigned int tNHits = 0; // number of hits (including virtual) of actual event

//	std::cout << "_checkForCorrelation "<<iRefHit<<" "<<iHit<<"\n";

	for (; iRefHit < rStopRefHitIndex && iHit < rStopHitIndex && iRefHit < nHits && iHit < nHits; ++iRefHit, ++iHit) {
		while (iRefHit < rStopRefHitIndex && iHit < rStopHitIndex && (rEventArray[iRefHit] + tEventNumberOffset) != rEventArray[iHit]) {  // reference hit and hit array are not in sync --> correct
			while (iRefHit < rStopRefHitIndex && ((rEventArray[iRefHit] + tEventNumberOffset) < rEventArray[iHit])) {  // hit array is at next event, catch up with reference hit array
				iRefHit++;
				tNrefHits++;
			}
			while (iHit < rStopHitIndex && ((rEventArray[iRefHit] + tEventNumberOffset) > rEventArray[iHit])) {  // reference hit array is at next event, catch up with hit array
				iHit++;
				tNHits++;
			}
//			std::cout<<"CHATCH UP "<<iRefHit<<"\t"<<iHit<<"\n";
		}

		if (iRefHit == rStopRefHitIndex || iHit == rStopHitIndex)  // one array is at the end, abort searching
			break;

		if (tRefEventNumber != rEventArray[iRefHit]) {  // new event trigger
			if (!tIsCorrelated) {
				if (tNVirtual != tNHits || tNVirtual != tNrefHits)  // if there are only virtual hits one cannot judge the correlation, do not reset good event counter
					tNgoodEvents = 0;
			}
			else
				tNgoodEvents++;

			if (tNgoodEvents >= nGoodEvents)
				return true;

			tRefEventNumber = rEventArray[iRefHit];
			tIsCorrelated = false;
			tNVirtual = 0;
		}

		if (rRefCol[iRefHit] != 0 && rCol[iHit] != 0 && rRefRow[iRefHit] != 0 && rRow[iHit] != 0 && std::fabs(rRefCol[iRefHit] - rCol[iHit]) < rError && std::fabs(rRefRow[iRefHit] - rRow[iHit]) < rError)  // check for correlation of real hits
			tIsCorrelated = true;
		if ((rRefCol[iRefHit] == 0 && rCol[iHit] == 0 && rRefRow[iRefHit] == 0 && rRow[iHit] == 0) || rCorrelated[iHit] == 0)  // if virtual hits occur in both devices correlation is likely
			tNVirtual++;
		if (_debug)
			std::cout << "\n" << iRefHit << "\t" << iHit << "\t" << rEventArray[iRefHit] << "\t" << rEventArray[iHit] << "\t" << rRefRow[iRefHit] << " / " << rRow[iHit] << "\t" << (int) rCorrelated[iHit] << "\t" << tNVirtual << "\t" << tIsCorrelated << "\t" << tNgoodEvents << "\n";
	}
	// Correct out of boundary indices
	if (iRefHit == rStopRefHitIndex || iRefHit == nHits)
		iRefHit--;
	if (iHit == rStopHitIndex || iHit == nHits)
		iHit--;
	return false;
}

bool _findCorrelation(unsigned int& iRefHit, unsigned int& iHit, const int64_t*& rEventArray, const double*& rRefCol, double*& rCol, const double*& rRefRow, double*& rRow, uint8_t*& rCorrelated, const unsigned int& nHits, const double& rError, const unsigned int& refArrayMaxIndex, unsigned int hitArraySearchRange, const unsigned int& nGoodEvents, const unsigned int& goodEventsSearchRange, bool setCorrelation = true)
{
	// Determine the search distance in the reference hit array
	unsigned int tStopRefHitIndex = nHits;
	if (iRefHit + refArrayMaxIndex < nHits)
		tStopRefHitIndex = iRefHit + refArrayMaxIndex;

	for (iRefHit; iRefHit < tStopRefHitIndex; ++iRefHit) {
		if (rRefCol[iRefHit] == 0 && rRefRow[iRefHit] == 0) {  // hit has to be non virtual (column/row != 0)
			if (setCorrelation)
				rCorrelated[iRefHit] |= 2;  // no match found
			continue;
		}

		if (_debug)
			std::cout << "Try to find hit for " << iRefHit << ": " << rEventArray[iRefHit] << " " << rRefCol[iRefHit] << " " << rRefRow[iRefHit] << "\n";

		// Determine the search distance for the correlated hit, search [iHit - hitArraySearchRange, iHit + hitArraySearchRange[

		unsigned int tStartHitIndex = 0;
		unsigned int tStopHitIndex = nHits;
		if (int(iRefHit - hitArraySearchRange) > 0)
			tStartHitIndex = iRefHit - hitArraySearchRange;
		if (iRefHit + hitArraySearchRange < nHits)
			tStopHitIndex = iRefHit + hitArraySearchRange;
		if (hitArraySearchRange == 0 && tStopHitIndex < nHits)  // special case (hitArraySearchRange == 0) to only search one hit
			tStopHitIndex = iRefHit + 1;
		if (_debug)
			std::cout << "Search between " << tStartHitIndex << " and " << tStopHitIndex << "\n";

		// Loop over the hits within the search distance and try to find a fitting hit. All fitting hits are checked to have subsequent correlated hits. Otherwise it is only correlation by chance.
		for (iHit = tStartHitIndex; iHit < tStopHitIndex; ++iHit) {
			if (rCol[iHit] == 0 && rCol[iRefHit] == 0)  //skip virtual hits
				continue;
			// Search for correlated hit candidate
			if (std::fabs(rRefCol[iRefHit] - rCol[iHit]) < rError && std::fabs(rRefRow[iRefHit] - rRow[iHit]) < rError) {  // check for correlation
				if (_debug)
					std::cout << "Try correlated hit canditate " << iHit << ": " << rEventArray[iHit] << " " << rCol[iHit] << " " << rRow[iHit] << "... ";
				if (_checkForCorrelation(iRefHit, iHit, rEventArray, rRefCol, rCol, rRefRow, rRow, rCorrelated, nHits, tStopRefHitIndex, iHit + goodEventsSearchRange, rError, nGoodEvents, true)) {  // correlated hit candidate is correct if 5 / 10 events are also correlated (including the candidate)
					if (_debug)
						std::cout << " SUCCESS! Is correlated hit!\n";
					// Goto first ref hit / hit of the actual correlated event
					unsigned int tiRefHit = iRefHit;
					unsigned int tiHit = iHit;
					for (; tiRefHit > 0 && rEventArray[tiRefHit] == rEventArray[iRefHit]; --tiRefHit) {  // mark all hits of last, incomplete event as not correlated
					}
					if (rEventArray[tiRefHit] != rEventArray[iRefHit])
						tiRefHit++;
					for (; tiHit > 0 && rEventArray[tiHit] == rEventArray[iHit]; --tiHit) {  // mark all hits of last, incomplete event as not correlated
					}
					if (rEventArray[tiHit] != rEventArray[iHit])
						tiHit++;
					iRefHit = tiRefHit;
					iHit = tiHit;
					return true;
				}
				else if (_debug)
					std::cout << "\n";
			}
		}
		if (_debug)
			std::cout << "No correlated hit for " << iRefHit << ": " << rEventArray[iRefHit] << " " << rRefCol[iRefHit] << " " << rRefRow[iRefHit] << "\n";
		if (setCorrelation)
			rCorrelated[iRefHit] |= 2;  // second bit shows uncorrelated hit (no match found)
	}
//	// Correct out of boundary indices FIME: this should be ok
//	if (iRefHit == nHits)
//		iRefHit--;
//	if (iHit == nHits)
//		iHit--;
	return false;
}

bool _fixAlignment(unsigned int iRefHit, unsigned int iHit, const int64_t*& rEventArray, const double*& rRefCol, double*& rCol, const double*& rRefRow, double*& rRow, const uint16_t*& rRefCharge, uint16_t*& rCharge, uint8_t*& rCorrelated, const unsigned int& nHits)
{
	if (_debug)
		std::cout << "Fix alignment " << iRefHit << ": " << rEventArray[iRefHit] << "/" << rRefCol[iRefHit] << "/" << rRefRow[iRefHit] << " = " << iHit << ": " << rEventArray[iHit] << "/" << rCol[iHit] << "/" << rRow[iHit] << "\n";
	int64_t tLastCopiedEvent = 0;  // last copied reference event number, needed to check if there all hits were copied
	int64_t tEventNumberOffset = rEventArray[iHit] - rEventArray[iRefHit];  // event number offset between reference hit and hit

	// negative offsets need temporary arrays for copying data
	unsigned int tHitIndex = iHit; // store start reference hit index for copying later
	double* tColCopy = 0;
	double* tRowCopy = 0;
	uint16_t* tChargeCopy = 0;
	uint8_t* tCorrelatedCopy = 0;
	if (tEventNumberOffset < 0) {
		tColCopy = new double[nHits];
		tRowCopy = new double[nHits];
		tChargeCopy = new uint16_t[nHits];
		tCorrelatedCopy = new uint8_t[nHits];
		for (unsigned int i = 0; i < nHits; ++i) {
			tColCopy[i] = 0;  // initialize as virtual hits only
			tRowCopy[i] = 0;  // initialize as virtual hits only
			tChargeCopy[i] = 0;  // initialize as virtual hits only
			tCorrelatedCopy[i] = rCorrelated[i];  // copy original
		}
	}

	for (; iRefHit < nHits && iHit < nHits; ++iRefHit, ++iHit) {
		while (iRefHit < nHits && iHit < nHits && (rEventArray[iRefHit] + tEventNumberOffset) != rEventArray[iHit]) {  // reference hit and hit array are not in sync --> correct
			//std::cout<<"Reference at "<<rEventArray[iRefHit]<<" hit at "<<rEventArray[iHit]<<"\n";
			while (iRefHit < nHits && ((rEventArray[iRefHit] + tEventNumberOffset) < rEventArray[iHit])) {  // hit array is at a next event, catch up with reference hit array
//				std::cout<<"Catch up reference array\n";
				rCol[iRefHit] = 0;
				rRow[iRefHit] = 0;
				rCharge[iRefHit] = 0;
				iRefHit++;
			}
			while (iHit < nHits && ((rEventArray[iRefHit] + tEventNumberOffset) > rEventArray[iHit])) {  // reference hit array is at a next event, catch up with hit array
				//				std::cout<<"Catch up hit array\n";
				if (rRow[iHit] != 0 && tLastCopiedEvent + tEventNumberOffset == rEventArray[iHit]) {  // true if not all real hits were copied -> mark all hits of this event as unsure correlated
					for (int tiRefHit = (int) iRefHit - 1; tiRefHit > 0; --tiRefHit) {  // mark all hits of last, incomplete event as unsure correlated (correlated = 0)
						if (tLastCopiedEvent == rEventArray[tiRefHit]){
							if (tEventNumberOffset > 0)
								rCorrelated[tiRefHit] = 0;
							else
								tCorrelatedCopy[tiRefHit] = 0;
						}
						else
							break;
					}
				}
				iHit++;
			}
		}

		if (iRefHit == nHits || iHit == nHits)  // one array is at the end, abort copying
			break;

		tLastCopiedEvent = rEventArray[iRefHit];

		while (rCol[iHit] == 0 && rRow[iHit] == 0 && iHit < nHits && ((rEventArray[iRefHit] + tEventNumberOffset) == rEventArray[iHit + 1]))  // do not copy virtual hits
			iHit++;

		if (_debug)
			std::cout << "\n" << iRefHit << "\t" << iHit << "\t" << rEventArray[iRefHit] << "\t" << rEventArray[iHit] << "\t" << rRefRow[iRefHit] << " / " << rRow[iHit] << "\n";

		// copy hits
		if (iHit < nHits) {
			if (tEventNumberOffset > 0) {
				rCol[iRefHit] = rCol[iHit];
				rRow[iRefHit] = rRow[iHit];
				rCharge[iRefHit] = rCharge[iHit];
				rCorrelated[iRefHit] = ((rCorrelated[iRefHit] & rCorrelated[iHit]) & 1);  // leave unsure correlation flag intact, no correlation flag (2nd bit set) is expected and reset
				rCol[iHit] = 0;
				rRow[iHit] = 0;
				rCharge[iHit] = 0;
			}
			else if (tEventNumberOffset < 0) {
				tColCopy[iRefHit] = rCol[iHit];
				tRowCopy[iRefHit] = rRow[iHit];
				tChargeCopy[iRefHit] = rCharge[iHit];
				tCorrelatedCopy[iRefHit] = ((rCorrelated[iRefHit] & rCorrelated[iHit]) & 1);  // leave unsure correlation flag intact, no correlation flag (2nd bit set) is expected and reset
//				std::cout << "rCorrelated[iHit] "<<(int) rCorrelated[iHit]<<"\n";
//				std::cout << "rCorrelated[iRefHit] "<<(int) rCorrelated[iRefHit]<<"\n";
//				std::cout << "tCorrelatedCopy[iRefHit] "<<(int) tCorrelatedCopy[iRefHit]<<"\n";
			}
		}
	}

	// Last events maybe do not exist in the hit array, thus set unsure correlation
	for (; iRefHit > 0 && iRefHit < nHits && rEventArray[iRefHit - 1] == rEventArray[iRefHit]; iRefHit++)
		;  // increase reference hit until first not copied event
	for (unsigned int i = iRefHit; i < nHits; ++i)
		rCorrelated[i] = 3;

	if (tEventNumberOffset < 0) {
		for (unsigned int i = tHitIndex; i < nHits; ++i) {  // copy results
			rCol[i] = tColCopy[i];
			rRow[i] = tRowCopy[i];
			rCharge[i] = tChargeCopy[i];
			rCorrelated[i] = tCorrelatedCopy[i];
		}
		delete[] tColCopy;
		delete[] tRowCopy;
		delete[] tChargeCopy;
		delete[] tCorrelatedCopy;
	}
	// Correct out of boundary indices
	if (iRefHit == nHits)
		iRefHit--;
	if (iHit == nHits)
		iHit--;

	return true;
}

void _mapCorrelationArray(const int64_t*& rEventArray, uint8_t*& rCorrelated, const unsigned int& nHits)
// correlation array is used to signal: 0 = unsure about correlation due to event merge where hits got lost, this cannot be fixed
//								     1 = event is correlated, start assumption
//									 2 = event hit has no corresponding hit, this flag is reset if hits are copied to this event
// correlation array is mapped to simple correlated (flag = 1) / uncorrelated (flag = 0, 2) array for the final result, one bad not correlated hit defines the complete event
{
	int64_t tEvent = rEventArray[0];
	uint8_t tEventCorrelation = 1;
	unsigned int i = 0;
	for (; i < nHits; ++i) {
		if (rEventArray[i] != tEvent){  // new event trigger
			if (tEventCorrelation != 1)
				tEventCorrelation = 0;
			for (int j = i - 1; j >= 0 && rEventArray[j] == tEvent; --j){
				rCorrelated[j] = tEventCorrelation;
			}
			tEvent = rEventArray[i];
			tEventCorrelation = 1;
		}
		tEventCorrelation &= (rCorrelated[i] == 1);
	}
	for (unsigned int j = i - 1; j > 0 && rEventArray[j] == tEvent; --j){
		rCorrelated[j] = tEventCorrelation;
	}
}

// Fix the event alignment with hit position information, crazy...
unsigned int fixEventAlignment(const int64_t*& rEventArray, const double*& rRefCol, double*& rCol, const double*& rRefRow, double*& rRow, const uint16_t*& rRefCharge, uint16_t*& rCharge, uint8_t*& rCorrelated, const unsigned int& nHits, const double& rError, const unsigned int& nBadEvents, const unsigned int& correltationSearchRange, const unsigned int& nGoodEvents, const unsigned int& goodEventsSearchRange)
{
	// Event number has to always increase, check here
	int64_t tEventNumber = rEventArray[0];
	for (unsigned int i = 0; i < nHits; ++i) {
		if (tEventNumber != rEventArray[i])
			if (tEventNumber > rEventArray[i]) {
				throw std::invalid_argument("The event number does not increase!");
			}
			else
				tEventNumber = rEventArray[i];
	}

	// traverse both hit arrays starting from 0
	unsigned int iRefHit = 0;  // actual reference hit array index
	unsigned int iHit = 0;  // actual hit array index
	unsigned int tNfixes = 0;  // number of fixes done
	bool tCorrelationBack = false;

	for (iRefHit; iRefHit < nHits; ++iRefHit) {
		iHit = iRefHit;
		if (_checkForNoCorrelation(iRefHit, iHit, rEventArray, rRefCol, rCol, rRefRow, rRow, rCorrelated, nHits, rError, nBadEvents)) { // false if all hits are correlated, nothing to do, thus return
			tCorrelationBack = false;  // assume that correlation does not come back
//			std::cout<<"BEFORE IT\n";
//			for (unsigned int i = 0; i < nHits; ++i){
//				std::cout << i  << "\t" << rEventArray[i] <<"\t" << rRefRow[i] << " / " << rRow[i] << "\t" << (int) rCorrelated[i] << "\n";
//			}
			if (_info)
				std::cout << "No correlation starting at index (event) " << iRefHit << " (" << rEventArray[iRefHit] << ") " << iHit << "\n";

			// Check if correlation comes back
			unsigned int tCorrBackRefHitIndex = iRefHit;
			unsigned int tCorrBackHitIndex = iHit;
			if (_findCorrelation(tCorrBackRefHitIndex, tCorrBackHitIndex, rEventArray, rRefCol, rCol, rRefRow, rRow, rCorrelated, nHits, rError, nHits, 0, nGoodEvents, goodEventsSearchRange, false)) {
				if (_info)
					std::cout << "But correlation is back at " << tCorrBackRefHitIndex << ": " << rRefRow[tCorrBackRefHitIndex] << " = " << tCorrBackHitIndex << ": " << rRow[tCorrBackHitIndex] << "\n";
				tCorrelationBack = true;  // the event alignment fixed itself
				if (iRefHit == tCorrBackRefHitIndex){
					std::cout << "ERROR "<<iRefHit<<" triggers correlation and no correlation... stop\n";
					break;
				}
			}
			else if (_info)
				std::cout << "Correlation for hit index > " << iRefHit << " comes never back...\n";

			if (!_findCorrelation(iRefHit, iHit, rEventArray, rRefCol, rCol, rRefRow, rRow, rCorrelated, tCorrBackRefHitIndex, rError, correltationSearchRange, correltationSearchRange, nGoodEvents, goodEventsSearchRange)) {
				if (_info)
					std::cout << "Found no correlation up to reference hit " << iRefHit - 1 << "\n";
				if (!tCorrelationBack) // if not correlation was found and the correlation does not come back abort, nothing can be done
					break;
			}
			else {
				if (iRefHit != iHit) {
					if (_info) {
						std::cout << "Start fixing correlation between " << iRefHit << " and " << tCorrBackRefHitIndex << "\n";
//						for (unsigned int i = 0; i < 10; ++i)
//							std::cout << "      fixing correlation for " << rEventArray[iRefHit + i] << ": " << rRefRow[iRefHit + i] << " = " << rEventArray[iHit + i]  << ": " << rRow[iHit + i] << "\n";
					}
					_fixAlignment(iRefHit, iHit, rEventArray, rRefCol, rCol, rRefRow, rRow, rRefCharge, rCharge, rCorrelated, tCorrBackRefHitIndex);
					tNfixes++;
//					return 0;
//					std::cout<<"FIXED IT\n";
//					for (unsigned int i = 0; i < nHits; ++i){
//						std::cout << i  << "\t" << rEventArray[i] <<"\t" << rRefRow[i] << " / " << rRow[i] << "\t" << (int) rCorrelated[i] << "\n";
//					}
				}
				else if (_info) {
					std::cout << "Correlation is back at " << iRefHit << ": " << rRefCol[iRefHit] << "/" << rRefRow[iRefHit] << " = " << iHit << ": " << rCol[iHit] << "/" << rRow[iHit] << "\n";
//					break;
				}
			}
		}
		else  // everything is correlated, nothing to do
			break;
	}

	_mapCorrelationArray(rEventArray, rCorrelated, nHits);
	return tNfixes;
}

