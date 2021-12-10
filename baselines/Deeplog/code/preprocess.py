import numpy  as np
import pandas as pd
import torch
from tqdm import tqdm

class Preprocessor(object):
    """Preprocessor for loading data from standard data formats."""

    def __init__(self, length, timeout, NO_EVENT=-1337):
        """Preprocessor for loading data from standard data formats.

            Parameters
            ----------
            length : int
                Number of events in context.

            timeout : float
                Maximum time between context event and the actual event in
                seconds.

            NO_EVENT : int, default=-1337
                ID of NO_EVENT event, i.e., event returned for context when no
                event was present. This happens in case of timeout or if an
                event simply does not have enough preceding context events.
            """
        # Set context length
        self.context_length = length
        self.timeout        = timeout

        # Set no-event event
        self.NO_EVENT = NO_EVENT

        # Set required columns
        self.REQUIRED_COLUMNS = {'timestamp', 'event', 'machine'}


    ########################################################################
    #                      General data preprocessing                      #
    ########################################################################

    def sequence(self, data, labels=None, verbose=False):
        """Transform pandas DataFrame into DeepCASE sequences.

            Parameters
            ----------
            data : pd.DataFrame
                Dataframe to preprocess.

            labels : int or array-like of shape=(n_samples,), optional
                If a int is given, label all sequences with given int. If an
                array-like is given, use the given labels for the data in file.
                Note: will overwrite any 'label' data in input file.

            verbose : boolean, default=False
                If True, prints progress in transforming input to sequences.

            Returns
            -------
            context : torch.Tensor of shape=(n_samples, context_length)
                Context events for each event in events.

            events : torch.Tensor of shape=(n_samples,)
                Events in data.

            labels : torch.Tensor of shape=(n_samples,)
                Labels will be None if no labels parameter is given, and if data
                does not contain any 'labels' column.

            mapping : dict()
                Mapping from new event_id to original event_id.
                Sequencing will map all events to a range from 0 to n_events.
                This is because event IDs may have large values, which is
                difficult for a one-hot encoding to deal with. Therefore, we map
                all Event ID values to a new value in that range and provide
                this mapping to translate back.
            """
        ################################################################
        #                  Transformations and checks                  #
        ################################################################

        # Case where a single label is given
        if isinstance(labels, int):
            # Set given label to all labels
            labels = np.full(data.shape[0], labels, dtype=int)

        # Transform labels to numpy array
        labels = np.asarray(labels)
        print(f'In preprocessor.py, sequence(), labels = {labels}, {labels.shape}')

        # Check if data contains required columns
        if set(data.columns) & self.REQUIRED_COLUMNS != self.REQUIRED_COLUMNS:
            raise ValueError(
                ".csv file must contain columns: {}"
                .format(list(sorted(self.REQUIRED_COLUMNS)))
            )

        # Check if labels is same shape as data
        if labels.ndim and labels.shape[0] != data.shape[0]:
            raise ValueError(
                "Number of labels: '{}' does not correspond with number of "
                "samples: '{}'".format(labels.shape[0], data.shape[0])
            )

        ################################################################
        #                          Map events                          #
        ################################################################

        # Create mapping of events
        mapping = {
            i: event for i, event in enumerate(np.unique(data['event'].values))
        }
        # mapping = {
        #     i: event for i, event in enumerate((data['event'].values))
        # }
        print(f"mapping = {len(mapping)}")

        # Check that NO_EVENT is not in events
        if self.NO_EVENT in mapping.values():
            raise ValueError(
                "NO_EVENT ('{}') is also a valid Event ID".format(self.NO_EVENT)
            )

        mapping[len(mapping)] = self.NO_EVENT
        mapping_inverse = {v: k for k, v in mapping.items()}

        # Apply mapping
        data['event'] = data['event'].map(mapping_inverse)

        ################################################################
        #                      Initialise results                      #
        ################################################################

        # Set events as events
        events = torch.Tensor(data['event'].values).to(torch.long)

        # Set context full of NO_EVENTs
        context = torch.full(
            size       = (data.shape[0], self.context_length),
            fill_value = mapping_inverse[self.NO_EVENT],
        ).to(torch.long)

        # Set labels if given
        if labels.ndim:
            labels = torch.Tensor(labels).to(torch.long)
        # Set labels if contained in data
        elif 'label' in data.columns:
            labels = torch.Tensor(data['label'].values).to(torch.long)
        # Otherwise set labels to None
        else:
            labels = None

        ################################################################
        #                        Create context                        #
        ################################################################

        # Sort data by timestamp
        data = data.sort_values(by='timestamp')

        # Group by machines
        machine_grouped = data.groupby('machine')
        # Add verbosity
        if verbose: machine_grouped = tqdm(machine_grouped, desc='Loading')

        # Group by machine
        for machine, events_ in machine_grouped:
            # Get indices, timestamps and events
            indices    = events_.index.values
            timestamps = events_['timestamp'].values
            events_    = events_['event'].values

            # Initialise context for single machine
            machine_context = np.full(
                (events_.shape[0], self.context_length),
                mapping_inverse[self.NO_EVENT],
                dtype = int,
            )

            # Loop over all parts of the context
            for i in range(self.context_length):

                # Compute time difference between context and event
                time_diff = timestamps[i+1:] - timestamps[:-i-1]
                # Check if time difference is larger than threshold
                timeout_mask = time_diff > self.timeout

                # Set mask to NO_EVENT
                machine_context[i+1:, self.context_length-i-1] = np.where(
                    timeout_mask,
                    mapping_inverse[self.NO_EVENT],
                    events_[:-i-1],
                )

            # Convert to torch Tensor
            machine_context = torch.Tensor(machine_context).to(torch.long)
            # Add machine_context to context
            context[indices] = machine_context

        ################################################################
        #                        Return results                        #
        ################################################################

        # Return result
        return context, events, labels, mapping


    ########################################################################
    #                     Preprocess different formats                     #
    ########################################################################

    def csv(self, path, nrows=None, labels=None, verbose=False):
        """Preprocess data from csv file.

            Note
            ----
            **Format**: The assumed format of a .csv file is that the first line
            of the file contains the headers, which should include
            ``timestamp``, ``machine``, ``event`` (and *optionally* ``label``).
            The remaining lines of the .csv file will be interpreted as data.

            Parameters
            ----------
            path : string
                Path to input file from which to read data.

            nrows : int, default=None
                If given, limit the number of rows to read to nrows.

            labels : int or array-like of shape=(n_samples,), optional
                If a int is given, label all sequences with given int. If an
                array-like is given, use the given labels for the data in file.
                Note: will overwrite any 'label' data in input file.

            verbose : boolean, default=False
                If True, prints progress in transforming input to sequences.

            Returns
            -------
            events : torch.Tensor of shape=(n_samples,)
                Events in data.

            context : torch.Tensor of shape=(n_samples, context_length)
                Context events for each event in events.

            labels : torch.Tensor of shape=(n_samples,)
                Labels will be None if no labels parameter is given, and if data
                does not contain any 'labels' column.
            """
        # Read data from csv file into pandas dataframe
        data = pd.read_csv(path, nrows=nrows)
        # assign labels 
        labels = data.labels 
        
        print(f'In preprocessor.py, csv(), the labels = {labels}')

        # Transform to sequences and return
        return self.sequence(data, labels=labels, verbose=verbose)