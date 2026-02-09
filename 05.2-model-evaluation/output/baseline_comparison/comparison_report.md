# Model Comparison: Fine-tuned vs. Mistral Instruct Baseline

Compared 15 samples

## Summary Statistics

| Metric | Fine-tuned | Instruct Baseline |
|--------|------------|-------------------|
| Stopped naturally (EOS) | 15/15 (100%) | 8/15 (53%) |
| Truncated (continuation) | 0/15 (0%) | 0/15 (0%) |
| Avg tokens generated | 53.9 | 105.2 |

---

## Sample 1

**Service:** CODEPIPELINE  
**Question Type:** comparison  
**Chunk ID:** codepipeline-faq-2

### Context
```
Section: AWS CodePipeline FAQs > Using AWS CodePipeline > How can I practice continuous delivery for my serverless applications and AWS Lambda functions?

You can release updates to your serverless application by including the
AWS Serverless Application Model
template and its corresponding files in your source code repository. You can use AWS CodeBuild in your pipeline to package your code for deployment. You can then use
AWS CloudFormation actions
to create a
change set
and deploy your serverless application. You have the option to extend your workflow with additional steps such as manual approvals or automated tests. Learn more
here
.

Using AWS CodePipeline and AWS CloudFormation, you can use continuous delivery to automatically build and test changes to your AWS CloudFormation stacks before promoting them to production stacks. This release process lets you rapidly and reliably make changes to your AWS infrastructure. You can extend your workflow with additional actions such as manual approvals, test actions, or invoke AWS Lambda actions. For more details, see
Continuous Delivery with AWS CloudFormation
page.

AWS CodePipeline integrates with AWS services such as AWS CodeCommit, Amazon S3, AWS CodeBuild, AWS CodeDeploy, AWS Elastic Beanstalk, AWS CloudFormation, AWS OpsWorks, Amazon ECS, and AWS Lambda. In addition, AWS CodePipeline integrates with a number of partner tools. For details see the
product integrations
page. Finally, you can write your own custom actions and i
...[truncated]
```

### Question
> How does AWS CodePipeline integrate with other AWS services?

### Reference Answer
AWS CodePipeline integrates with various AWS services including AWS CodeCommit, Amazon S3, AWS CodeBuild, AWS CodeDeploy, AWS Elastic Beanstalk, AWS CloudFormation, AWS OpsWorks, Amazon ECS, and AWS Lambda. It also integrates with partner tools and allows for the creation of custom actions to integrate any existing tool with CodePipeline.

### Fine-tuned Model Answer
AWS CodePipeline integrates with AWS services such as AWS CodeCommit, Amazon S3, AWS CodeBuild, AWS CodeDeploy, AWS Elastic Beanstalk, AWS CloudFormation, AWS OpsWorks, Amazon ECS, and AWS Lambda. Additionally, it integrates with a number of partner tools, and users can write their own custom actions and integrate any existing tool with CodePipeline.

*Tokens: 84, Stopped naturally: ✓, Truncated: ✗*

### Instruct Baseline Answer
AWS CodePipeline integrates with various AWS services such as AWS CodeCommit, Amazon S3, AWS CodeBuild, AWS CodeDeploy, AWS Elastic Beanstalk, AWS CloudFormation, AWS OpsWorks, Amazon ECS, and AWS Lambda. These integrations allow you to create continuous delivery pipelines that automatically build, test, and deploy your code changes across different AWS services. Additionally, AWS CodePipeline integrates with a number of partner tools, and you can write your own custom actions to integrate any existing tool with CodePipeline. For more details, see the product integrations page.

*Tokens: 125, Stopped naturally: ✓, Truncated: ✗*

### Manual Assessment

**Fine-tuned:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Instruct Baseline:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Which is better?** [ ] Fine-tuned / [ ] Instruct / [ ] Tie

**Notes:**

---

## Sample 2

**Service:** CERTIFICATE  
**Question Type:** comparison  
**Chunk ID:** certificate-faq-5

### Context
```
Section: AWS Certificate Manager FAQs > ACM Certificates > Can ACM provide certificates with multiple domain names?

Yes. Each certificate must include at least one domain name, and you can add additional names to the certificate if you want to. For example, you can add the name “www.example.net” to a certificate for “www.example.com” if users can reach your site by either name. You must own or control all of the names included in your certificate request.

A wildcard domain name matches any first level subdomain or hostname in a domain. A first-level subdomain is a single domain name label that does not contain a period (dot). For example, you can use the name *.example.com to protect www.example.com, images.example.com, and any other host name or first-level subdomain that ends with .example.com. Learn more in the
ACM User Guide
.

Certificates issued through ACM are valid for 13 months (395 days). If you issue private certificates directly from a private CA and manage the keys and certificates without using ACM for certificate management, you can choose any validity period, including an absolute end date or a relative time that is days, months, or years from the present time.

Yes, AWS Certificate Manager (ACM) will provide shorter validity lifetimes for public certificates to align with the Certificate Authority/Browser Forum (CA/Browser Forum) requirements for TLS certificates. ACM already offers automatic handling of certificate renewals and capabilities to notify you w
...[truncated]
```

### Question
> How does a wildcard domain name differ from a regular domain name in terms of subdomain matching?

### Reference Answer
A wildcard domain name matches any first-level subdomain or hostname in a domain, while a regular domain name does not have this capability. For example, a wildcard name like *.example.com can protect multiple first-level subdomains such as www.example.com and images.example.com, whereas a regular domain name would only represent itself specifically.

### Fine-tuned Model Answer
A wildcard domain name matches any first level subdomain or hostname in a domain, meaning it can protect multiple subdomains under that domain. In contrast, a regular domain name only protects the specific domain name it is associated with.

*Tokens: 50, Stopped naturally: ✓, Truncated: ✗*

### Instruct Baseline Answer
A wildcard domain name is a special type of domain name that matches any first-level subdomain or hostname in a domain. A first-level subdomain is a single domain name label that does not contain a period (dot). For example, you can use the name *.example.com to protect www.example.com, images.example.com, and any other host name or first-level subdomain that ends with .example.com.

A regular domain name, on the other hand, only matches the exact domain name that is specified. For example, a certificate for example.com would only protect

*Tokens: 128, Stopped naturally: ✗, Truncated: ✗*

### Manual Assessment

**Fine-tuned:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Instruct Baseline:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Which is better?** [ ] Fine-tuned / [ ] Instruct / [ ] Tie

**Notes:**

---

## Sample 3

**Service:** EC2  
**Question Type:** comparison  
**Chunk ID:** ec2-faq-9

### Context
```
Section: Amazon EC2 FAQs > General > EC2 On-Demand Instance limits

API will no longer return the max-instances value. Instead you can now use the Service Quotas APIs to retrieve information about EC2 limits. You can find more information about the Service Quotas APIs in the
AWS documentation
.
Q: Will the vCPU limits have an impact on my monthly bill?
No. EC2 usage is still calculated either by the hour or the second,
depending on which AMI you're running
and the instance type and size you’ve launched.
Q: Will vCPU limits be available in all Regions?
vCPU-based instance limits are available in all commercial AWS Regions.


```

### Question
> Are vCPU limits available in all AWS Regions, and if so, which type of AWS Regions?

### Reference Answer
Yes, vCPU-based instance limits are available in all commercial AWS Regions.

### Fine-tuned Model Answer
vCPU-based instance limits are available in all commercial AWS Regions.

*Tokens: 16, Stopped naturally: ✓, Truncated: ✗*

### Instruct Baseline Answer
Yes, vCPU-based instance limits are available in all commercial AWS Regions.

*Tokens: 18, Stopped naturally: ✓, Truncated: ✗*

### Manual Assessment

**Fine-tuned:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Instruct Baseline:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Which is better?** [ ] Fine-tuned / [ ] Instruct / [ ] Tie

**Notes:**

---

## Sample 4

**Service:** QUICKSIGHT  
**Question Type:** comparison  
**Chunk ID:** quicksight-faq-0

### Context
```
Analytics
›
Amazon QuickSight
›
FAQs

General
5
Authors and Readers
5
Mobile and web access
2
Data management
8
User management
1
Visualization and analysis
9
Security and access
6
Sharing
1
Upgrades and downgrades
2

Mobile and web access

Visualization and analysis

Upgrades and downgrades

Amazon QuickSight is a unified business intelligence service that makes it easier for all employees within an organization to build visualizations, perform ad hoc analysis, and quickly get business insights from their data, anytime, on any device. With QuickSight, all users can meet varying analytic needs from the same source of truth through modern interactive dashboards, paginated reports, natural language queries, and embedded analytics. With Amazon Q in QuickSight, business analysts and business users can use natural language to build, discover, and share meaningful insights in seconds, turning insights into impact faster. Upload CSV and Excel files; connect to SaaS applications like Salesforce; access on-premises databases like SQL Server, MySQL, and PostgreSQL; and seamlessly discover your AWS data sources such as Amazon Redshift, Amazon Relational Database Service (Amazon RDS), Amazon Aurora, Amazon Athena, and Amazon Simple Storage Service (Amazon S3). QuickSight enables organizations to scale their business analytics capabilities to hundreds of thousands of users, and delivers fast and responsive query performance by using a robust in-memory engine: SPICE.


```

### Question
> In what ways do Amazon QuickSight's capabilities support fast query performance compared to traditional methods?

### Reference Answer
Amazon QuickSight supports fast and responsive query performance through its robust in-memory engine called SPICE. This allows it to deliver quick insights and analytics capabilities to a large number of users, which is a significant advantage over traditional methods that may not utilize such an efficient in-memory architecture.

### Fine-tuned Model Answer
Amazon QuickSight delivers fast and responsive query performance by using a robust in-memory engine called SPICE, which allows for the scaling of business analytics capabilities to hundreds of thousands of users. This contrasts with traditional methods, which may not be as efficient or scalable.

*Tokens: 58, Stopped naturally: ✓, Truncated: ✗*

### Instruct Baseline Answer
Amazon QuickSight utilizes a robust in-memory engine called SPICE (Super-fast, Parallel, In-memory Calculation Engine) to deliver fast and responsive query performance. SPICE is designed to handle large datasets and complex analytical queries, allowing users to get answers in seconds. This is in contrast to traditional methods that may rely on slower disk-based storage or less efficient query processing, leading to longer wait times for results. Additionally, QuickSight's data model is optimized for querying, allowing for efficient data access and analysis. Overall, these features enable QuickSight to provide faster query

*Tokens: 128, Stopped naturally: ✗, Truncated: ✗*

### Manual Assessment

**Fine-tuned:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Instruct Baseline:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Which is better?** [ ] Fine-tuned / [ ] Instruct / [ ] Tie

**Notes:**

---

## Sample 5

**Service:** KINESIS  
**Question Type:** comparison  
**Chunk ID:** kinesis-faq-3

### Context
```
Section: Amazon Kinesis Data Streams FAQs > Key concepts > What is a partition key?

A partition key is used to isolate and route records to different shards of a data stream. A partition key is specified by your data producer while adding data to a Kinesis data stream. For example, let’s say you have a data stream with two shards (shard 1 and shard 2). You can configure your data producer to use two partition keys (key A and key B) so that all records with key A are added to shard 1 and all records with key B are added to shard 2.

A sequence number is a unique identifier for each record. Sequence number is assigned by Amazon Kinesis when a data producer calls PutRecord or PutRecords operation to add data to a Amazon Kinesis data stream. Sequence numbers for the same partition key generally increase over time; the longer the time period between
PutRecord
or
PutRecords
requests, the larger the sequence numbers become.

The capacity mode of Kinesis Data Streams determines how capacity is managed and usage is charged for a data stream. You can choose between provisioned and on-demand modes. In provisioned mode, you specify the number of shards for the data stream. The total capacity of a data stream is the sum of the capacities of its shards. You can increase or decrease the number of shards in a data stream as needed, and you pay for the number of shards at an hourly rate. In on-demand mode, AWS manages the shards to provide the necessary throughput. You pay only for the actua
...[truncated]
```

### Question
> What is the difference between provisioned mode and on-demand mode in Kinesis Data Streams regarding cost structure?

### Reference Answer
In provisioned mode, you pay for the number of shards at an hourly rate, while in on-demand mode, you only pay for the actual throughput used. Provisioned mode requires you to specify the number of shards, whereas on-demand mode allows AWS to manage the necessary throughput without user intervention.

### Fine-tuned Model Answer
In provisioned mode, you specify the number of shards for the data stream and pay for the number of shards at an hourly rate, while in on-demand mode, you pay only for the actual throughput used, and Kinesis Data Streams automatically accommodates your workload throughput needs as they ramp up or down.

*Tokens: 72, Stopped naturally: ✓, Truncated: ✗*

### Instruct Baseline Answer
In provisioned mode, you pay a fixed hourly rate for the number of shards you provision in your data stream. This means that you have a consistent throughput capacity, and you are charged even if the actual data throughput is lower. On the other hand, in on-demand mode, you pay only for the actual data throughput used, with no upfront costs or minimum fees. This means that you can scale your data stream to handle varying data throughput needs, and you only pay for the resources you consume.

In summary, provisioned mode offers a consistent throughput capacity at a fixed cost,

*Tokens: 128, Stopped naturally: ✗, Truncated: ✗*

### Manual Assessment

**Fine-tuned:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Instruct Baseline:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Which is better?** [ ] Fine-tuned / [ ] Instruct / [ ] Tie

**Notes:**

---

## Sample 6

**Service:** KMS  
**Question Type:** conceptual  
**Chunk ID:** kms-faq-22

### Context
```
Section: AWS Key Management Service FAQs > External key store > What risks do I accept if I choose XKS instead of using standard KMS keys generated and stored in AWS KMS?

Availability risk:
You are responsible for the availability of the XKS Proxy and external key material. This system must have high availability to verify that whenever you need an XKS key to decrypt an encrypted resource or encrypt new data, AWS KMS can successfully connect to the XKS proxy, which itself can connect to your external key manager to complete the necessary cryptographic operation using the key. For example, suppose you encrypted an EBS volume using an XKS key and now you want to launch an EC2 instance and attach that encrypted volume. The EC2 service will pass the unique encrypted data key for that volume to AWS KMS to decrypt it so it can be provisioned in volatile memory of the Nitro card in order to decrypt and encrypt read/write operations to the volume. If your XKS Proxy or external key manager isn’t available to decrypt the volume key, your EC2 instance will fail to launch. In these types of failures, AWS KMS returns a KMSInvalidStateException stating that the XKS Proxy is not available. It is now up to you to determine why your XKS Proxy and key manager is unavailable based on the error messages provided by KMS.
Durability risk:
Because keys are under your control in systems outside of AWS, you are solely responsible for the durability of all external keys you create. If the external ke
...[truncated]
```

### Question
> Why is it important to maintain the availability of the XKS Proxy and external key manager?

### Reference Answer
Maintaining the availability of the XKS Proxy and external key manager is crucial because if they are unavailable, AWS KMS cannot perform the necessary cryptographic operations to decrypt or encrypt data. This can lead to failures, such as the inability to launch an EC2 instance that requires decryption of an encrypted EBS volume, resulting in a KMSInvalidStateException error.

### Fine-tuned Model Answer
It is important to maintain the availability of the XKS Proxy and external key manager because if they are not available, you will not be able to decrypt the XKS key, which can lead to failures in services such as EC2 instances not launching.

*Tokens: 54, Stopped naturally: ✓, Truncated: ✗*

### Instruct Baseline Answer
It is important to maintain the availability of the XKS Proxy and external key manager because they are necessary for AWS KMS to perform cryptographic operations using XKS keys. If the XKS Proxy or external key manager is not available, AWS KMS will be unable to decrypt or encrypt data using the XKS keys, which can result in various issues such as EC2 instances failing to launch or encrypted data becoming unrecoverable. Therefore, ensuring the high availability of the XKS Proxy and external key manager is crucial for maintaining the security and functionality of your encrypted data in AWS.

*Tokens: 124, Stopped naturally: ✓, Truncated: ✗*

### Manual Assessment

**Fine-tuned:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Instruct Baseline:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Which is better?** [ ] Fine-tuned / [ ] Instruct / [ ] Tie

**Notes:**

---

## Sample 7

**Service:** VPC  
**Question Type:** conceptual  
**Chunk ID:** vpc-faq-10

### Context
```
Section: Amazon VPC FAQs > IP Address Manager > What is VPC IP Address Manager (IPAM)?

Amazon VPC IP Address Manager (IPAM) is a managed service that makes it easier for you to plan, track, and monitor IP addresses for your AWS workloads. Using IPAM, you can easily organize IP addresses based on your routing and security needs and set simple business rules to govern IP address assignments. You can also automate IP address assignment to VPCs, eliminating the need to use spreadsheet-based or homegrown IP address planning applications, which can be hard to maintain and time-consuming. IPAM provides a unified operational view, which can be used as your single source of truth, enabling you to quickly and efficiently perform routine IP address management activities such as tracking IP utilization, troubleshooting, and auditing.

You should use IPAM to make IP address management more efficient. Existing mechanisms that leverage spreadsheets or home-grown tools require manual work, and are error-prone. With IPAM, as an example, you can roll out applications faster as your developers no longer need to wait for the central IP address administration team to allocate IP addresses. You can also detect overlapping IP addresses and fix them before there is a network outage. In addition, you can create alarms for IPAM to notify you if the address pools are nearing exhaustion or if resources fail to comply with allocations rules set on a pool. These are some of the many reasons you should us
...[truncated]
```

### Question
> How does IPAM enhance the efficiency of IP address management compared to traditional methods?

### Reference Answer
IPAM enhances efficiency by eliminating the need for spreadsheet-based or homegrown IP address planning applications, which are often hard to maintain and time-consuming. It automates IP address assignments, allowing developers to roll out applications faster without having to wait for central IP address administration teams. Additionally, IPAM can detect overlapping IP addresses and set alarms for address pool exhaustion, which reduces manual work and minimizes errors.

### Fine-tuned Model Answer
IPAM enhances efficiency by automating IP address assignment, eliminating the need for manual work and spreadsheet-based or homegrown IP address planning applications, which can be hard to maintain and time-consuming. It also provides a unified operational view, enabling quick and efficient performance of routine IP address management activities.

*Tokens: 67, Stopped naturally: ✓, Truncated: ✗*

### Instruct Baseline Answer
IPAM enhances the efficiency of IP address management compared to traditional methods in several ways:

1. Automated IP address allocation: IPAM can automate IP address allocations across multiple accounts and VPCs based on configurable business rules, eliminating the need for manual intervention and reducing the time and effort required for IP address management.
2. Centralized IP address management: IPAM provides a unified operational view, which can be used as a single source of truth for IP address management, enabling you to quickly and efficiently perform routine IP address management activities such as tracking IP utilization, troubleshooting, and

*Tokens: 128, Stopped naturally: ✗, Truncated: ✗*

### Manual Assessment

**Fine-tuned:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Instruct Baseline:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Which is better?** [ ] Fine-tuned / [ ] Instruct / [ ] Tie

**Notes:**

---

## Sample 8

**Service:** ELASTICACHE  
**Question Type:** conceptual  
**Chunk ID:** elasticache-faq-36

### Context
```
Section: Amazon ElastiCache FAQs > Global Datastore > What is ElastiCache Global Datastore?

Global Datastore is a feature of ElastiCache
that provides fully managed, fast, reliable and security-focused cross-Region replication. With Global Datastore, you can write to your cache in one Region and have the data available for read in up to two other cross-Region replica clusters, thereby enabling low-latency reads and disaster recovery across Regions.
Designed for real-time applications with a global footprint, Global Datastore typically replicates data across Regions in under one second, increasing the responsiveness of your applications by providing geolocal reads closer to end users. In the unlikely event of Regional degradation, one of the healthy cross-Region replica caches can be promoted to become the primary with full read and write capabilities. Once initiated, the promotion typically completes in less than a minute, allowing your applications to remain available.

You can replicate to up to two secondary Regions within a Global Datastore. The caches in secondary Regions can be used to serve low-latency local reads and for disaster recovery in the unlikely event of Regional degradation.

Global Datastore is supported on ElastiCache for Redis 5.0.6 onward.

No, ElastiCache doesn’t automatically promote a secondary cluster in the event when a primary cluster (Region) is degraded. You can manually initiate the failover by promoting a secondary cluster to become a primary.
...[truncated]
```

### Question
> How does Global Datastore enhance the performance of real-time applications with a global footprint?

### Reference Answer
Global Datastore enhances the performance of real-time applications with a global footprint by replicating data across Regions in under one second, which increases application responsiveness by providing geolocal reads closer to end users.

### Fine-tuned Model Answer
Global Datastore enhances the performance of real-time applications with a global footprint by replicating data across Regions in under one second, which increases the responsiveness of applications by providing geolocal reads closer to end users.

*Tokens: 50, Stopped naturally: ✓, Truncated: ✗*

### Instruct Baseline Answer
Global Datastore enhances the performance of real-time applications with a global footprint by providing low-latency reads closer to end users. It achieves this by replicating data across Regions in under one second, allowing for geolocal reads from the secondary clusters in the closer Region to the user. In the event of a Regional degradation, one of the healthy cross-Region replica caches can be promoted to become the primary, ensuring that the applications remain available with minimal downtime.

*Tokens: 106, Stopped naturally: ✓, Truncated: ✗*

### Manual Assessment

**Fine-tuned:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Instruct Baseline:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Which is better?** [ ] Fine-tuned / [ ] Instruct / [ ] Tie

**Notes:**

---

## Sample 9

**Service:** STORAGEGATEWAY  
**Question Type:** conceptual  
**Chunk ID:** storagegateway-faq-27

### Context
```
Section: AWS Storage Gateway FAQs > Hardware Appliance > How do I use the hardware appliance?

You configure an IP address through the local hardware appliance console, and use that IP address in the AWS Storage Gateway console to activate your appliance. This associates your hardware appliance with your AWS account. Once the hardware appliance is activated, you create your desired gateway type to run on the hardware appliance. The selected type of gateway is then enabled on the appliance. You manage and use your Storage Gateway Hardware Appliance from the AWS Console, similar to how you would with the virtual appliance. For more information, please see the
hardware appliance documentation
.

No. The hardware appliance supports running only one gateway at a time.

Yes. To change the gateway type after it is installed on a hardware appliance, you choose
Remove Gateway
from the Storage Gateway console, which deletes the gateway and all associated resources. At that point, you are free to launch a new gateway on the hardware appliance.

You can add more storage to the 5 TB model to increase the usable local cache capacity to 12 TB by adding 5 additional SSDs to the appliance. If you have already activated the appliance and associated it with your AWS account, you will need to factory reset it before adding more storage.

No. Only add the SSDs that are available from the manufacturer of the appliance. These SSDs are qualified for use in the Storage Gateway Hardware Appliance.

Ye
...[truncated]
```

### Question
> What happens if you want to change the gateway type after it has been installed on a hardware appliance?

### Reference Answer
If you want to change the gateway type after it has been installed on a hardware appliance, you need to choose 'Remove Gateway' from the Storage Gateway console. This action deletes the existing gateway and all associated resources, allowing you to launch a new gateway on the hardware appliance.

### Fine-tuned Model Answer
To change the gateway type after it is installed on a hardware appliance, you must choose 'Remove Gateway' from the Storage Gateway console, which deletes the gateway and all associated resources. Once this is done, you can launch a new gateway on the hardware appliance.

*Tokens: 61, Stopped naturally: ✓, Truncated: ✗*

### Instruct Baseline Answer
To change the gateway type after it has been installed on a hardware appliance, you need to remove the existing gateway first. You can do this by choosing "Remove Gateway" from the Storage Gateway console, which deletes the gateway and all associated resources. After that, you are free to launch a new gateway of the desired type on the hardware appliance.

*Tokens: 79, Stopped naturally: ✓, Truncated: ✗*

### Manual Assessment

**Fine-tuned:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Instruct Baseline:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Which is better?** [ ] Fine-tuned / [ ] Instruct / [ ] Tie

**Notes:**

---

## Sample 10

**Service:** VPN  
**Question Type:** conceptual  
**Chunk ID:** vpn-faq-10

### Context
```
Section: AWS VPN FAQs > AWS Client VPN authentication and authorization > How do I use security group to restrict access to my applications for only Client VPN connections?

For your application, you can specify to allow access only from the security groups that were applied to the associated subnet. Now you limit access to only users connected via Client VPN.

Yes, you can upload a new metadata document in the IAM identity provider associated with the Client VPN endpoint. Updated metadata are reflected in 2 to 4 hours.

No, you must use the AWS Client VPN software client to connect to the endpoint.

Client VPN exports the connection log as a best effort to CloudWatch logs. These logs are exported periodically at 15 minute intervals. The connection logs include details on created and terminated connection requests.

No. You can use Amazon VPC Flow Logs in the associated VPC.

Yes, using the CLI or console, you can view the current active connections for an endpoint and terminate active connections.

Yes. Using CloudWatch monitor you can see Ingress and Egress bytes and Active connections for each Client VPN Endpoint.

The software client for AWS Client VPN is compatible with existing AWS Client VPN configurations. The client supports adding profiles using the OpenVPN configuration file generated by the AWS Client VPN service. Once the profile is created, the client will connect to your endpoint based on your settings.

The software client is provided free of charge. You will 
...[truncated]
```

### Question
> What are the requirements for installing the AWS Client VPN software client on Windows and Mac?

### Reference Answer
You need admin access to install the app on both Windows and Mac. After the installation, admin access is not required.

### Fine-tuned Model Answer
You need admin access to install the app on both Windows and Mac. After that point, admin access is not required.

*Tokens: 25, Stopped naturally: ✓, Truncated: ✗*

### Instruct Baseline Answer
To install the AWS Client VPN software client on Windows and Mac, you will need the following:

For Windows:
- 64-bit Windows 10
- Admin access for installation

For Mac:
- macOS (Mojave, Catalina, and Big Sur)
- Admin access for installation

After the installation, admin access is no longer required. The client supports the OpenVPN protocol and all the features provided by the AWS Client VPN service. It is provided free of charge, and you will only be billed for AWS Client VPN service usage. IT administrators

*Tokens: 128, Stopped naturally: ✗, Truncated: ✗*

### Manual Assessment

**Fine-tuned:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Instruct Baseline:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Which is better?** [ ] Fine-tuned / [ ] Instruct / [ ] Tie

**Notes:**

---

## Sample 11

**Service:** CLOUDTRAIL  
**Question Type:** factual  
**Chunk ID:** cloudtrail-faq-13

### Context
```
Section: AWS CloudTrail FAQs > CloudTrail Lake > What additional information can I add to my CloudTrail management and data events?

You can enrich CloudTrail events with resource tags on AWS Resources- By incorporating your organization's resource tagging strategy into your CloudTrail events, you can more easily categorize and analyze AWS activities in the context of your business operations, projects, or departments. For example, let's say you use resource tags to mark your production S3 buckets containing critical data. You can now easily view all CloudTrail events matching these specific tags as this information is included in the event itself. No more manual cross-referencing across multiple systems to find this information.
IAM Global Condition Keys provide another way to add context to your CloudTrail events is through keys, including Principal Tags. When enabled, CloudTrail will include information about AWS condition keys that were evaluated during the authorization process. This can provide additional details about the principal making the request and specifics about the request itself- for example, you can view the aws:SourceAccount, for API calls made to your resource directly by an
AWS service principal
. It's important to note that a condition key will only appear in the enriched event if it was evaluated as part of the IAM policy during the authorization process.
You can enrich your CloudTrail events when setting up CloudTrail Lake event data stores. During the
...[truncated]
```

### Question
> What feature allows you to enrich CloudTrail events with resource tags on AWS Resources?

### Reference Answer
You can enrich CloudTrail events with resource tags on AWS Resources by incorporating your organization's resource tagging strategy into your CloudTrail events, which helps categorize and analyze AWS activities in the context of your business operations, projects, or departments.

### Fine-tuned Model Answer
You can enrich CloudTrail events with resource tags on AWS Resources by incorporating your organization's resource tagging strategy into your CloudTrail events. This allows you to more easily categorize and analyze AWS activities in the context of your business operations, projects, or departments.

*Tokens: 58, Stopped naturally: ✓, Truncated: ✗*

### Instruct Baseline Answer
The feature that allows you to enrich CloudTrail events with resource tags on AWS Resources is the use of resource tagging strategy in CloudTrail events. By incorporating your organization's resource tagging strategy into your CloudTrail events, you can more easily categorize and analyze AWS activities in the context of your business operations, projects, or departments. This information is included in the event itself, eliminating the need for manual cross-referencing across multiple systems to find this information.

*Tokens: 101, Stopped naturally: ✓, Truncated: ✗*

### Manual Assessment

**Fine-tuned:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Instruct Baseline:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Which is better?** [ ] Fine-tuned / [ ] Instruct / [ ] Tie

**Notes:**

---

## Sample 12

**Service:** S3  
**Question Type:** factual  
**Chunk ID:** s3-faq-31

### Context
```
Section: Amazon S3 FAQs > S3 Access Grants > Are there any quotas for S3 Access Grants?

Yes. You can create up to 100,000 grants per S3 Access Grants instance, and up to 1,000 locations per S3 Access Grants instance.

No. The latency for obtaining temporary credentials from S3 Access Grants is similar to obtaining temporary credentials from AWS STS today. Once you have obtained the credentials from S3 Access Grants, you can reuse unexpired credentials for subsequent requests. For these subsequent requests, there is no additional latency for requests authenticated via S3 Access Grants credentials compared to other methods.

If you intend to use S3 Access Grants for directory identities, you will need to set up AWS IAM Identity Center first. AWS IAM Identity Center helps you create or connect your workforce identities, whether the identities are created and stored in Identity Center, or in an external third-party Identity Provider. Refer to the
Identity Center documentation
for the setup process. Once you have set up the Identity Center instance, you can connect the instance to S3 Access Grants. Thereafter, S3 Access Grants relies on Identity Center to retrieve user attributes such as group membership to evaluate requests and make authorization decisions.

Yes. Whereas today, you initialize your S3 client with IAM credentials associated with your application (for example, IAM role credentials for EC2 or IAM Roles Anywhere; or using long-term IAM user credentials), your applica
...[truncated]
```

### Question
> What is the maximum number of grants and locations you can create per S3 Access Grants instance?

### Reference Answer
You can create up to 100,000 grants and up to 1,000 locations per S3 Access Grants instance.

### Fine-tuned Model Answer
You can create up to 100,000 grants per S3 Access Grants instance, and up to 1,000 locations per S3 Access Grants instance.

*Tokens: 41, Stopped naturally: ✓, Truncated: ✗*

### Instruct Baseline Answer
You can create up to 100,000 grants and up to 1,000 locations per S3 Access Grants instance.

*Tokens: 33, Stopped naturally: ✓, Truncated: ✗*

### Manual Assessment

**Fine-tuned:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Instruct Baseline:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Which is better?** [ ] Fine-tuned / [ ] Instruct / [ ] Tie

**Notes:**

---

## Sample 13

**Service:** CLOUDFRONT  
**Question Type:** factual  
**Chunk ID:** cloudfront-faq-12

### Context
```
Section: Amazon CloudFront FAQs > Saas Manager > Who should use CloudFront SaaS Manager?

CloudFront SaaS Manager is designed for organizations that face the challenge of managing multiple websites efficiently. Software-as-a-Service (SaaS) and web development platform providers will find it particularly valuable, as it allows them to maintain consistent settings across their tenants' websites. Similarly, businesses managing multiple corporate websites can use it to standardize their web presence while preserving the flexibility to customize individual sites. If you only have  a handful of websites or each website contains different CloudFront configurations, then the single tenant (traditional) distribution is likely a better fit.

CloudFront is available as an option via the AWS console and APIs for customers who wish to manage shared settings across groups of domains. Here’s how to get started: 1/Define shared settings: Create a multi-tenant distribution which contains shared settings that will act as the template for groups of domains. 2/Create distribution tenants: Create distribution tenants that allow you to associate domains and their TLS certificate to a multi-tenant distribution. 3/Fine- tune control: Optionally, customize settings for distribution tenants by applying overrides.

A multi-tenant distribution defines the base configuration that will be shared across domains. It contains shared configuration settings such as origin configurations, cache behaviors, and s
...[truncated]
```

### Question
> What is the purpose of CloudFront SaaS Manager?

### Reference Answer
CloudFront SaaS Manager is designed for organizations that face the challenge of managing multiple websites efficiently. It is particularly valuable for Software-as-a-Service (SaaS) and web development platform providers, as it allows them to maintain consistent settings across their tenants' websites.

### Fine-tuned Model Answer
CloudFront SaaS Manager is designed for organizations that face the challenge of managing multiple websites efficiently. It allows software-as-a-service (SaaS) and web development platform providers to maintain consistent settings across their tenants' websites, as well as businesses managing multiple corporate websites to standardize their web presence while preserving flexibility for individual sites.

*Tokens: 73, Stopped naturally: ✓, Truncated: ✗*

### Instruct Baseline Answer
CloudFront SaaS Manager is a feature of Amazon CloudFront designed to help organizations efficiently manage multiple websites. It is particularly useful for Software-as-a-Service (SaaS) and web development platform providers, as well as businesses managing multiple corporate websites, by allowing them to maintain consistent settings across their tenants' websites while preserving the flexibility to customize individual sites. It enables defining shared settings, creating distribution tenants, and fine-tuning control for each tenant. The multi-tenant distribution defines the base configuration that will be shared across domains, and each distribution tenant inherits this base configuration and can include custom

*Tokens: 128, Stopped naturally: ✗, Truncated: ✗*

### Manual Assessment

**Fine-tuned:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Instruct Baseline:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Which is better?** [ ] Fine-tuned / [ ] Instruct / [ ] Tie

**Notes:**

---

## Sample 14

**Service:** Q  
**Question Type:** factual  
**Chunk ID:** q-faq-1

### Context
```
Section: Amazon Q – Generative AI Assistant > Amazon Q in AWS Services > Generative AI assistance for business intelligence

Amazon Q brings advanced generative AI technology to
Amazon QuickSight
, the AWS unified business intelligence (BI) service built for the cloud. With Amazon Q in QuickSight, your business analysts get a generative BI assistant they can use to build BI dashboards, visualizations, and complex calculations in minutes, all while using natural language. Empower any employee to create customizable data stories and go beyond dashboard-delivered insights with multi-visual Q&A responses and AI-driven executive summaries. Discover key insights, trends, and drivers with agentic AI to make smarter decisions across the business.

Amazon Q adds generative AI capabilities in
Amazon Connect
, the AWS cloud contact center, to support contact center agents and end-customers. Amazon Q in Connect automatically detects customer issues from the conversation and leverages generative AI to deliver real-time, personalized responses and recommended actions, leveraging customer information and content from your knowledge repositories and external website. Reduce handle times, deliver exceptional customer service, and lower service costs thanks to your AI virtual assistant for customer service.

Amazon Q introduces advanced generative AI capabilities in
AWS Supply Chain
, our cloud-based supply chain management application. It helps you operate your supply chain more efficiently b
...[truncated]
```

### Question
> What functionality does Amazon Q provide within Amazon QuickSight?

### Reference Answer
Amazon Q in QuickSight provides a generative BI assistant that allows business analysts to build BI dashboards, visualizations, and complex calculations quickly using natural language. It empowers employees to create customizable data stories and offers multi-visual Q&A responses and AI-driven executive summaries to discover key insights, trends, and drivers.

### Fine-tuned Model Answer
Amazon Q in QuickSight allows business analysts to build BI dashboards, visualizations, and complex calculations in minutes using natural language. It empowers employees to create customizable data stories and go beyond dashboard-delivered insights with multi-visual Q&A responses and AI-driven executive summaries.

*Tokens: 68, Stopped naturally: ✓, Truncated: ✗*

### Instruct Baseline Answer
Amazon Q provides advanced generative AI technology within Amazon QuickSight, enabling business analysts to build BI dashboards, visualizations, and complex calculations in minutes using natural language. It empowers any employee to create customizable data stories, go beyond dashboard-delivered insights with multi-visual Q&A responses, and receive AI-driven executive summaries. This helps discover key insights, trends, and drivers to make smarter business decisions.

*Tokens: 96, Stopped naturally: ✓, Truncated: ✗*

### Manual Assessment

**Fine-tuned:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Instruct Baseline:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Which is better?** [ ] Fine-tuned / [ ] Instruct / [ ] Tie

**Notes:**

---

## Sample 15

**Service:** AUDIT  
**Question Type:** factual  
**Chunk ID:** audit-faq-4

### Context
```
Section: AWS Audit Manager FAQs > Getting started > How do I get started with AWS Audit Manager?

You can get started by setting up AWS Audit Manager in the AWS Management Console, AWS CLI, or via API. AWS Audit Manager documentation contains a
getting started tutorial
, which provides a hands-on introduction to AWS Audit Manager. In this tutorial, you can create an assessment using a standard framework and begin the automated collection of evidence.

Yes, AWS Audit Manager supports multiple accounts via integration with AWS Organizations. AWS Audit Manager and AWS Organizations integration enables you to run an AWS Audit Manager assessment over multiple accounts and consolidate evidence into a delegated administrator account.

You can specify the scope by selecting the AWS accounts when you launch an assessment from a framework. The framework used defines the AWS services from which AWS Audit Manager collects evidence.

AWS Audit Manager saves you time by automatically collecting and organizing evidence as defined by each control requirement. With Audit Manager, you can focus on reviewing the relevant evidence to ensure your controls are working as intended. When it is time for an audit, AWS Audit Manager helps you manage stakeholder reviews of your controls and enables you to build audit-ready reports with much less manual effort. For example, the delegation feature enables you to assign controls in your assessment to a subject matter expert to review. After reviewing and s
...[truncated]
```

### Question
> How can you get started with AWS Audit Manager?

### Reference Answer
You can get started by setting up AWS Audit Manager in the AWS Management Console, AWS CLI, or via API. Additionally, the AWS Audit Manager documentation contains a getting started tutorial that provides a hands-on introduction to AWS Audit Manager, allowing you to create an assessment using a standard framework and begin the automated collection of evidence.

### Fine-tuned Model Answer
You can get started with AWS Audit Manager by setting up AWS Audit Manager in the AWS Management Console, AWS CLI, or via API.

*Tokens: 31, Stopped naturally: ✓, Truncated: ✗*

### Instruct Baseline Answer
To get started with AWS Audit Manager, you can set up an assessment using a standard framework in the AWS Management Console, AWS CLI, or via API. AWS Audit Manager documentation provides a getting started tutorial that walks you through creating an assessment and beginning the automated collection of evidence. AWS Audit Manager also supports multiple accounts through integration with AWS Organizations, allowing you to run assessments over multiple accounts and consolidate evidence into a delegated administrator account. You can specify the scope of the assessment by selecting the AWS accounts and services when launching an assessment from a framework. AWS Audit Manager saves you time by automatically collecting

*Tokens: 128, Stopped naturally: ✗, Truncated: ✗*

### Manual Assessment

**Fine-tuned:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Instruct Baseline:**
- [ ] Factually correct
- [ ] No hallucinations
- [ ] Complete answer
- [ ] Follows instruction (answer only from context)

**Which is better?** [ ] Fine-tuned / [ ] Instruct / [ ] Tie

**Notes:**

---

## Overall Assessment

### Patterns Observed

**Fine-tuned model strengths:**
- 

**Fine-tuned model weaknesses:**
- 

**Instruct baseline strengths:**
- 

**Instruct baseline weaknesses:**
- 

### Win Rate

Count manually after filling in assessments:
- Fine-tuned wins: ___
- Instruct wins: ___
- Ties: ___

### Recommendation

Based on this comparison:
- [ ] Fine-tuning provides significant value → Deploy fine-tuned model
- [ ] Marginal improvement → Consider if fine-tuning effort is worth it
- [ ] No improvement or worse → Use Instruct baseline, investigate why fine-tuning didn't help
