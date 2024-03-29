import express from "express";
import bodyParser from "body-parser";
import mongoose from "mongoose";
import cors from 'cors';
import dotenv from 'dotenv';
import postsRoutes from './routes/posts.js';
import userRoutes from './routes/users.js';

const app = express();
dotenv.config();
app.use(bodyParser.json({limit: "30mb", extended: true}));
app.use(bodyParser.urlencoded({limit: "30mb", extended: true}));
app.use(cors());

app.use('/posts', postsRoutes);
app.use('/user', userRoutes); // for user authentication
// const CONNECTION_URL = 'mongodb+srv://djsurt:djsurt123@cluster0.i7gyzkp.mongodb.net/?retryWrites=true&w=majority';
const PORT = process.env.PORT || 4004;

mongoose.connect(process.env.CONNECTION_URL, {useNewUrlParser: true, useUnifiedTopology: true})
                .then(()=> app.listen(PORT, ()=> console.log(`Server running on port: ${PORT}`)))
                .catch((error)=>{console.log(error.message)});

//mongoose.set('useFindAndModify', false);
