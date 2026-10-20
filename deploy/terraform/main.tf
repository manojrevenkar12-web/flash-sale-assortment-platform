# AWS resources for the serving side and the batch pods: a container registry, the data bucket
# (landing to gold, encrypted, versioned) and IAM roles for the Kubernetes service accounts
# (IRSA) on an existing EKS cluster. Databricks workspaces and EKS itself are managed elsewhere.
terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.80" }
  }
  backend "s3" {}
}

provider "aws" {
  region = var.region
  default_tags { tags = { project = "flash-sale-assortment", environment = var.environment } }
}

variable "region" {
  type    = string
  default = "eu-central-1"
}

variable "environment" {
  type    = string
  default = "prod"
}

variable "eks_cluster_name" {
  type = string
}

locals {
  name = "fsap-${var.environment}"
}

data "aws_eks_cluster" "this" {
  name = var.eks_cluster_name
}

data "aws_iam_openid_connect_provider" "eks" {
  url = data.aws_eks_cluster.this.identity[0].oidc[0].issuer
}

resource "aws_ecr_repository" "app" {
  name                 = "flash-sale-assortment-platform"
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_kms_key" "data" {
  description         = "${local.name} data"
  enable_key_rotation = true
}

resource "aws_s3_bucket" "data" {
  bucket_prefix = "${local.name}-data-"
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.data.arn
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    id     = "expire-bronze-and-quarantine"
    status = "Enabled"
    filter {
      prefix = "bronze/"
    }
    expiration {
      days = 90
    }
  }
  rule {
    id     = "expire-quarantine"
    status = "Enabled"
    filter {
      prefix = "quarantine/"
    }
    expiration {
      days = 30
    }
  }
}

data "aws_iam_policy_document" "assume" {
  for_each = toset(["fsap-api", "fsap-batch"])
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [data.aws_iam_openid_connect_provider.eks.arn]
    }
    condition {
      test     = "StringEquals"
      variable = "${replace(data.aws_eks_cluster.this.identity[0].oidc[0].issuer, "https://", "")}:sub"
      values   = ["system:serviceaccount:assortment:${each.key}"]
    }
  }
}

resource "aws_iam_role" "sa" {
  for_each           = data.aws_iam_policy_document.assume
  name               = "${local.name}-${trimprefix(each.key, "fsap-")}"
  assume_role_policy = each.value.json
}

data "aws_iam_policy_document" "api" {
  statement {
    actions   = ["s3:GetObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.data.arn, "${aws_s3_bucket.data.arn}/gold/*", "${aws_s3_bucket.data.arn}/reports/*"]
  }
  statement {
    actions   = ["kms:Decrypt"]
    resources = [aws_kms_key.data.arn]
  }
}

data "aws_iam_policy_document" "batch" {
  statement {
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
    resources = [aws_s3_bucket.data.arn, "${aws_s3_bucket.data.arn}/*"]
  }
  statement {
    actions   = ["kms:Decrypt", "kms:GenerateDataKey"]
    resources = [aws_kms_key.data.arn]
  }
}

resource "aws_iam_role_policy" "api" {
  role   = aws_iam_role.sa["fsap-api"].id
  policy = data.aws_iam_policy_document.api.json
}

resource "aws_iam_role_policy" "batch" {
  role   = aws_iam_role.sa["fsap-batch"].id
  policy = data.aws_iam_policy_document.batch.json
}

output "ecr_repository_url" {
  value = aws_ecr_repository.app.repository_url
}

output "data_bucket" {
  value = aws_s3_bucket.data.bucket
}

output "role_arns" {
  value = { for k, r in aws_iam_role.sa : k => r.arn }
}
