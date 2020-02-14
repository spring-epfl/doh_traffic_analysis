# Datasets



Overview of the datasets (folder names are the same as the identifiers used in the paper):

| Folder name             | Location  | Resolver   | Client     | Platform      | Paper Section used in |
| ----------------------- | --------- | ---------- | ---------- | ------------- | --------------------- |
|                         |           |            |            |               |                       |
| LOC1                    | Lausanne  | Cloudflare | Cloudflare | Desktop       | Section 5             |
| LOC2                    | Leuven    | Cloudflare | Cloudflare | Desktop       | Section 5             |
| LOC3                    | Singapore | Cloudflare | Cloudflare | Desktop (AWS) | Section 5             |
| OW                      | Lausanne  | Cloudflare | Cloudflare | Desktop       | Section 5             |
| RPI                     | Lausanne  | Cloudflare | Cloudflare | Raspberry Pi  | Section 5             |
| CL-FF                   | Leuven    | Cloudflare | Cloudflare | Desktop       | Section 5             |
| GOOGLE                  | Leuven    | Google     | Firefox    | Desktop       | Section 5             |
| CLOUD                   | Leuven    | Cloudflare | Firefox    | Desktop       | Section 5             |
| EDNS0-128               | Lausanne  | Cloudflare | Cloudflare | Desktop       | Section 6             |
| EDNS0-468               | Lausanne  | Cloudflare | Cloudflare | Desktop       | Section 6             |
| TOR                     | Lausanne  | Cloudflare | Cloudflare | Desktop       | Section 6             |
| EDNS0-128-adblock       | Lausanne  | Cloudflare | Cloudflare | Desktop       | Section 6             |
| countermeasure_overhead | Lausanne  | Cloudflare | Cloudflare | Desktop       | Section 6             |



### Format

Filenames in the folder are of the format: **dd**-**mm**-**yy**-**HHMMSS**_**MacID.json**, where **MacID** is the ID of the machine on which the experiment was run. Older datasets (LOC1 and LOC2), have files of the format **dd-mm-yy_MacID.json**.

The JSON file is of the format:

```
{
	N.pcap : 
	{
		sent: [],
		received: [],
		order: []
	}
}
```

where:

- N.pcap -- N indicates the index of the website in the list of queried websites (see code/collection/short_list_1500 for an example)
- sent -- an array of TLS record sizes, sent from the client to the resolver.
- received -- an array of TLS record sizes, received by the client from the resolver.
- order -- an array of +1 and -1 values, where +1 indicates outgoing (from client to resolver), and -1 indicates incoming (from resolver to client) records.

### Raw data

Please note that this is processed data, where TLS record information has been extracted from PCAP files. If you require access to the raw PCAP files, please get in touch with [sandra.siby@epfl.ch]() or marc.juarez@usc.edu.

